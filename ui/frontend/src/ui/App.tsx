import React, {useEffect, useMemo, useRef, useState} from 'react'
import {openProject,listScenes,listSegments,applyChanges,save,rebuildScene, listCandidates, listOrigSegments} from '../api'

const BACKEND_BASE = `${window.location.protocol}//${window.location.hostname}:8787`

// 格式化时间为 时:分:秒.毫秒 格式
function formatTime(seconds: number): string {
  if (typeof seconds !== 'number' || isNaN(seconds)) return '0:00:00.000'
  
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)
  const milliseconds = Math.floor((seconds % 1) * 1000)
  
  return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${milliseconds.toString().padStart(3, '0')}`
}

type Scene = {clip_scene_id:number,count:number,avg_conf:number,chain_len:number}

type SegmentRow = {
  seg_id: number
  clip: { scene_seg_idx?: number, start?: number, end?: number, scene_id?: number }
  matched_orig_seg?: any
  top_matches?: any[]
}

export default function App(){
  const [root,setRoot]=useState(''); const [movie,setMovie]=useState(''); const [clip,setClip]=useState('')

  // Load last inputs from localStorage on first render
  useEffect(() => {
    try {
      const r = localStorage.getItem('rm_root');
      const m = localStorage.getItem('rm_movie');
      const c = localStorage.getItem('rm_clip');
      if (r) setRoot(r);
      if (m) setMovie(m);
      if (c) setClip(c);
    } catch {}
  }, []);

  const [scenes,setScenes]=useState<Scene[]>([])
  const [activeScene,setActiveScene]=useState<number|null>(null)
  const [segments,setSegments]=useState<SegmentRow[]>([])
  const [allSegments,setAllSegments]=useState<SegmentRow[]>([])
  const [selectedSegId,setSelectedSegId]=useState<number|null>(null)
  const [selectedCandIdx,setSelectedCandIdx]=useState<number>(0)
  const [followMovie,setFollowMovie]=useState<boolean>(true)
  const [loop,setLoop]=useState<boolean>(true)
  const [mirrorClip, setMirrorClip] = useState<boolean>(false)
  const [candMode, setCandMode] = useState<'top'|'scene'|'all'>('top')
  const [candList, setCandList] = useState<any[]>([])
  const [origSegments, setOrigSegments] = useState<any[]>([])
  const [showOrigSegments, setShowOrigSegments] = useState<boolean>(false)

  const clipRef = useRef<HTMLVideoElement|null>(null)
  const movieRef = useRef<HTMLVideoElement|null>(null)

  const clipLoopHandlerRef = useRef<any>(null)
  const movieLoopHandlerRef = useRef<any>(null)

  const [debug, setDebug] = useState<boolean>(true)
  const clipLastLogRef = useRef<number>(0)
  const movieLastLogRef = useRef<number>(0)
  const clipSeekingRef = useRef<boolean>(false)
  const movieSeekingRef = useRef<boolean>(false)
  const dlog = (...args:any[]) => { if (debug) console.log('[UI]', ...args) }

  const [showDebugPanel, setShowDebugPanel] = useState<boolean>(true)
  const [overrides, setOverrides] = useState<any>(null)
  const [sceneHints, setSceneHints] = useState<Record<number, {scene_id:number, scene_seg_idx:number} | null>>({})

  async function refreshOverrides(){
    try{
      const resp = await fetch(`${BACKEND_BASE}/overrides`)
      const j = await resp.json()
      setOverrides(j)
      dlog('overrides', j)
    }catch(e){ console.error('load overrides failed', e) }
  }
  async function clearOverrides(){
    try{
      await fetch(`${BACKEND_BASE}/overrides/clear`, {method:'POST'})
      await refreshOverrides()
    }catch(e){ console.error('clear overrides failed', e) }
  }

  async function ensureSceneHint(sceneId: number){
    if (sceneHints.hasOwnProperty(sceneId)) return
    try{
      const arr = await listSegments(sceneId)
      let hint: any = null
      if (Array.isArray(arr) && arr.length){
        const row = arr[0] as any
        const mo = row?.matched_orig_seg || (row?.top_matches && row.top_matches[0])
        if (mo && mo.scene_id!=null && mo.scene_seg_idx!=null){
          hint = { scene_id: mo.scene_id, scene_seg_idx: mo.scene_seg_idx }
        }
      }
      setSceneHints(prev=> ({ ...prev, [sceneId]: hint }))
    }catch(e){ console.error('ensureSceneHint failed', e) }
  }
  useEffect(()=>{
    if (!Array.isArray(scenes) || scenes.length===0) return
    const firstIds = scenes.slice(0, Math.min(20, scenes.length)).map(s=>s.clip_scene_id)
    firstIds.forEach(id=>{ ensureSceneHint(id) })
  }, [scenes])

  const [clipSrc, setClipSrc] = useState<string>(`${BACKEND_BASE}/video/clip`)
  const [movieSrc, setMovieSrc] = useState<string>(`${BACKEND_BASE}/video/movie`)
  
  // 播放器同步和进度条状态
  const [syncPlay, setSyncPlay] = useState<boolean>(true)
  const [clipCurrentTime, setClipCurrentTime] = useState<number>(0)
  const [clipDuration, setClipDuration] = useState<number>(0)
  const [movieCurrentTime, setMovieCurrentTime] = useState<number>(0)
  const [movieDuration, setMovieDuration] = useState<number>(0)
  const [isPlaying, setIsPlaying] = useState<boolean>(false)
  
  // 当前播放的segment信息
  // 播放状态管理
  const [playingSegId, setPlayingSegId] = useState<number | null>(null)
  const [playingFromSide, setPlayingFromSide] = useState<'left' | 'right' | null>(null) // 记录播放来源
  const [playingType, setPlayingType] = useState<'clip' | 'orig' | null>(null) // 记录播放类型
  const [playingOrigSegId, setPlayingOrigSegId] = useState<number | null>(null) // 记录正在播放的原始segment ID
  const [currentPlayingTimeRange, setCurrentPlayingTimeRange] = useState<{
    clipStart: number,
    clipEnd: number, 
    movieStart: number,
    movieEnd: number
  } | null>(null) // 记录当前播放的时间范围
  
  // 当前校对的clip segment信息
  const [currentClipSegment, setCurrentClipSegment] = useState<{
    segId: number,
    clipStart: number,
    clipEnd: number,
    sceneId?: number,
    sceneSegIdx?: number
  } | null>(null) // 记录当前正在校对的clip segment信息
  
  // 循环播放计数管理
  const [loopCount, setLoopCount] = useState<number>(0) // 当前已循环次数
  const [maxLoops, setMaxLoops] = useState<number>(3) // 最大循环次数
  const loopCountRef = useRef<number>(0) // 用于在事件处理器中访问最新的循环计数
  
  // 同步 loopCountRef 和 loopCount 状态
  useEffect(() => {
    loopCountRef.current = loopCount
  }, [loopCount])

  async function refreshScenes(){
    try {
      const arr: Scene[] = await listScenes()
      setScenes(arr)
      
      // 加载所有场景的段落，按 seg_id 展开显示
      const allSegs: SegmentRow[] = []
      for (const scene of arr) {
        try {
          const segs = await listSegments(scene.clip_scene_id)
          allSegs.push(...segs)
        } catch (e) {
          console.error(`Failed to load segments for scene ${scene.clip_scene_id}`, e)
        }
      }
      setAllSegments(allSegs)
      
      if (allSegs && allSegs.length) {
        const firstSeg = allSegs[0]
        setSelectedSegId(firstSeg.seg_id)
        // 初始化当前校对的clip segment信息
        setCurrentClipSegment({
          segId: firstSeg.seg_id,
          clipStart: firstSeg.clip.start ?? 0,
          clipEnd: firstSeg.clip.end ?? 0,
          sceneId: firstSeg.clip.scene_id,
          sceneSegIdx: firstSeg.clip.scene_seg_idx
        })
      }
    } catch (e) {
      console.error('listScenes failed', e)
    }
  }

  async function doOpen(){
    try {
      await openProject(root, movie||undefined, clip||undefined)
    } catch (e) {
      console.error('openProject failed', e)
    }
    await refreshScenes()
    await refreshOverrides()
  }

  // load segments when selection changes
  useEffect(()=>{
    if(selectedSegId != null){
      // 当选择了 seg_id 时，从 allSegments 中找到对应的段落
      const selectedSeg = allSegments.find(s => s.seg_id === selectedSegId)
      if (selectedSeg) {
        // 设置当前活动场景为该段落所属的场景
        setActiveScene(selectedSeg.clip.scene_id || null)
        
        // 加载该场景的所有段落
        if (selectedSeg.clip.scene_id != null) {
          listSegments(selectedSeg.clip.scene_id).then(arr=>{
            setSegments(arr)
            loadCandidates(selectedSegId, candMode)
          }).catch(e => {
            console.error('Failed to load segments', e)
          })
        }
      }
    }
  },[selectedSegId, allSegments])

  // fetch candidates for current seg
  async function loadCandidates(segId: number, mode: 'top'|'scene'|'all'){
    try{
      const resp = await listCandidates(segId, mode, 50, 0)
      const items = resp?.items || []
      setCandList(items)
      setSelectedCandIdx(0)
    }catch(e){
      console.error('listCandidates failed', e)
      const r = allSegments.find(s=>s.seg_id===segId) || segments.find(s=>s.seg_id===segId)
      setCandList(r?.top_matches || [])
      setSelectedCandIdx(0)
    }
  }

  useEffect(()=>{
    if (selectedSegId!=null) loadCandidates(selectedSegId, candMode)
  }, [selectedSegId, candMode])

  // 加载原始段落数据 (当前场景及前后场景)
  async function loadOrigSegments(sceneId: number) {
    try {
      const promises = []
      const sceneIds = []
      
      // 加载前两个场景
      for (let i = sceneId - 2; i <= sceneId + 2; i++) {
        if (i > 0) { // 只加载有效的场景ID
          promises.push(listOrigSegments(i))
          sceneIds.push(i)
        }
      }
      
      const responses = await Promise.all(promises)
      
      // 合并所有场景的segments，并添加场景标识
      const allSegments = []
      responses.forEach((resp, idx) => {
        const segments = resp.segments || []
        segments.forEach((seg: any) => {
          allSegments.push({
            ...seg,
            _sceneId: sceneIds[idx], // 添加场景标识
            _isCurrentScene: sceneIds[idx] === sceneId // 标识是否为当前场景
          })
        })
      })
      
      // 按场景ID和segment索引排序
      allSegments.sort((a, b) => {
        if (a._sceneId !== b._sceneId) {
          return a._sceneId - b._sceneId
        }
        return (a.scene_seg_idx || 0) - (b.scene_seg_idx || 0)
      })
      
      setOrigSegments(allSegments)
    } catch (e) {
      console.error('Failed to load orig segments', e)
      setOrigSegments([])
    }
  }

  // derive selected row
  const selectedRow: SegmentRow | undefined = useMemo(()=>{
    return allSegments.find(s=>s.seg_id===selectedSegId) || segments.find(s=>s.seg_id===selectedSegId)
  },[allSegments, segments, selectedSegId])

  // 当选择"场景内"模式时加载原始段落
  useEffect(() => {
    if (candMode === 'scene' && selectedRow?.matched_orig_seg?.scene_id) {
      loadOrigSegments(selectedRow.matched_orig_seg.scene_id)
      setShowOrigSegments(true)
    } else {
      setShowOrigSegments(false)
    }
  }, [candMode, selectedRow])

  function seekWhenReady(v: HTMLVideoElement | null, t: number){
    if (!v) return
    const doSeek = () => { try { v.currentTime = Math.max(t, 0); dlog('set currentTime', t, 'ready=', v.readyState) } catch {} }
    if (v.readyState >= 1) {
      doSeek()
    } else {
      dlog('wait loadedmetadata before seek to', t)
      const onMeta = () => { v.removeEventListener('loadedmetadata', onMeta); doSeek() }
      v.addEventListener('loadedmetadata', onMeta)
    }
  }

  function attachLoopSafe(v: HTMLVideoElement | null, s: number, e: number, ref: React.MutableRefObject<any>, seekingRef: React.MutableRefObject<boolean>){
    if (!v) return
    if (ref.current) {
      v.removeEventListener('timeupdate', ref.current)
      ref.current = null
    }
    const handler = () => {
      if (!loop) return
      if (seekingRef?.current) return
      const span = e - s
      if (span > 0.05 && v.currentTime > e - 0.03) {
        // 检查是否达到最大循环次数
        if (loopCountRef.current >= maxLoops) {
          // 达到最大循环次数，停止播放
          try { 
            v.pause()
          } catch {}
          return
        }
        
        // 继续循环并增加计数
        try { 
          v.currentTime = s 
          loopCountRef.current += 1
          setLoopCount(loopCountRef.current)
        } catch {}
      }
    }
    v.addEventListener('timeupdate', handler)
    ref.current = handler
  }

  // 统一的播放控制函数
  function playSegment(segmentData: {
    segId: number,
    clipStart: number,
    clipEnd: number,
    movieStart: number,
    movieEnd: number,
    fromSide: 'left' | 'right',
    type: 'clip' | 'orig',
    origSegId?: number  // 可选的原始segment ID
  }) {
    const { segId, clipStart, clipEnd, movieStart, movieEnd, fromSide, type, origSegId } = segmentData
    
    dlog('playSegment', { segId, clipStart, clipEnd, movieStart, movieEnd, fromSide, type, origSegId })

    const cv = clipRef.current
    const mv = movieRef.current

    // Build URLs with explicit time start
    const clipUrl = `${BACKEND_BASE}/video/clip?t=${clipStart.toFixed(3)}`
    const movieUrl = `${BACKEND_BASE}/video/movie?t=${movieStart.toFixed(3)}`
    
    setClipSrc(clipUrl)
    setMovieSrc(movieUrl)

    // For streams, loop bounds should be relative to 0
    const clipLoopStart = 0
    const clipLoopEnd = Math.max(0.01, (clipEnd - clipStart))
    const movieLoopStart = 0
    const movieLoopEnd = Math.max(0.01, (movieEnd - movieStart))

    // Seek to start of streams (0 for ?t= streams)
    seekWhenReady(cv, 0)
    seekWhenReady(mv, 0)

    // Set up loop handlers
    attachLoopSafe(cv, clipLoopStart, clipLoopEnd, clipLoopHandlerRef, clipSeekingRef)
    attachLoopSafe(mv, movieLoopStart, movieLoopEnd, movieLoopHandlerRef, movieSeekingRef)
    
    // 更新播放状态
    setPlayingSegId(segId)
    setPlayingFromSide(fromSide)
    setPlayingType(type)
    
    // 如果有原始segment ID，设置它
    if (origSegId !== undefined) {
      setPlayingOrigSegId(origSegId)
    } else {
      setPlayingOrigSegId(null)
    }
    
    setCurrentPlayingTimeRange({
      clipStart,
      clipEnd,
      movieStart,
      movieEnd
    })
    
    // 重置循环计数
    setLoopCount(0)
    loopCountRef.current = 0
    
    // 播放控制
    if (syncPlay) {
      // 在同步模式下，使用统一的播放控制
      setTimeout(() => {
        handlePlay()
      }, 100)
    } else {
      // 非同步模式，独立播放
      const tryPlay = (video: HTMLVideoElement | null) => {
        if (!video) return
        const attemptPlay = () => {
          try { 
            video.play().catch(()=>{})
          } catch {}
        }
        
        if (video.readyState >= 3) { // HAVE_FUTURE_DATA
          attemptPlay()
        } else {
          const onCanPlay = () => {
            video.removeEventListener('canplay', onCanPlay)
            attemptPlay()
          }
          video.addEventListener('canplay', onCanPlay)
          // Fallback timeout
          setTimeout(attemptPlay, 300)
        }
      }
      
      tryPlay(cv)
      tryPlay(mv)
    }
  }

  // Seek videos to the currently selected row (clip uses clip times, movie uses current choice/candidate)
  function seekTo(row?: SegmentRow, candIdx?: number){
    const r = row || selectedRow
    if (!r) return
    const clipStart = r.clip.start ?? 0
    const clipEnd = r.clip.end ?? (clipStart + 2)
    // candidate or current matched selection
    let mo = r.matched_orig_seg || {}
    const cand = (candList && candList[candIdx ?? selectedCandIdx]) ||
                 ((r.top_matches && r.top_matches[candIdx ?? selectedCandIdx]) || null)
    if (followMovie){
      if (cand) mo = cand
    }
    const movStart = mo?.start ?? 0
    const movEnd = mo?.end ?? (movStart + 2)

    playSegment({
      segId: r.seg_id,
      clipStart,
      clipEnd,
      movieStart: movStart,
      movieEnd: movEnd,
      fromSide: 'left',
      type: 'clip'
    })
  }

  // Seek to a specific original segment without affecting candidate state
  function seekToOrigSegment(origSeg: any) {
    if (!origSeg || !currentClipSegment) return
    
    // 对于场景内的segments，左侧应该播放当前校对的clip segment
    // 而不是寻找对应的clip segment
    const clipStart = currentClipSegment.clipStart
    const clipEnd = currentClipSegment.clipEnd
    
    const movStart = origSeg.start ?? 0
    const movEnd = origSeg.end ?? (movStart + 2)
    
    playSegment({
      segId: currentClipSegment.segId, // 使用当前校对的clip segment ID，而不是origSeg.seg_id
      clipStart,
      clipEnd,
      movieStart: movStart,
      movieEnd: movEnd,
      fromSide: 'right',
      type: 'orig',
      origSegId: origSeg.seg_id  // 传递原始segment ID
    })
  }

  // 专门用于右侧候选项列表的播放
  function seekToCandidate(candidate: any, candIdx: number) {
    if (!currentClipSegment || !candidate) return
    
    // 使用当前校对的clip segment信息
    const clipStart = currentClipSegment.clipStart
    const clipEnd = currentClipSegment.clipEnd
    
    // 使用候选项的movie信息
    const movStart = candidate.start ?? 0
    const movEnd = candidate.end ?? (movStart + 2)
    
    // 更新候选项选择状态
    setSelectedCandIdx(candIdx)
    
    playSegment({
      segId: currentClipSegment.segId, // 使用当前校对的clip segment ID
      clipStart,
      clipEnd,
      movieStart: movStart,
      movieEnd: movEnd,
      fromSide: 'right',
      type: 'clip'
    })
  }

  // 同步播放控制函数
  function handlePlay() {
    const cv = clipRef.current
    const mv = movieRef.current
    
    if (syncPlay && cv && mv) {
      // 确保两个视频都准备好了再同步播放
      const playWhenReady = async () => {
        // 等待两个视频都有足够的数据
        const waitForReady = (video: HTMLVideoElement) => {
          return new Promise<void>((resolve) => {
            if (video.readyState >= 3) { // HAVE_FUTURE_DATA
              resolve()
            } else {
              const onCanPlay = () => {
                video.removeEventListener('canplay', onCanPlay)
                resolve()
              }
              video.addEventListener('canplay', onCanPlay)
            }
          })
        }
        
        try {
          // 等待两个视频都准备好
          await Promise.all([waitForReady(cv), waitForReady(mv)])
          
          // 同时开始播放
          await Promise.all([
            cv.play().catch(()=>{}),
            mv.play().catch(()=>{})
          ])
          
          setIsPlaying(true)
        } catch (e) {
          console.error('Failed to sync play:', e)
          // 降级到普通播放
          cv.play().catch(()=>{})
          mv.play().catch(()=>{})
          setIsPlaying(true)
        }
      }
      
      playWhenReady()
    } else if (cv || mv) {
      // 非同步模式，直接播放
      cv?.play().catch(()=>{})
      mv?.play().catch(()=>{})
      setIsPlaying(true)
    }
  }

  function handlePause() {
    const cv = clipRef.current
    const mv = movieRef.current
    if (syncPlay && cv && mv) {
      // 同时暂停
      cv.pause()
      mv.pause()
    } else {
      cv?.pause()
      mv?.pause()
    }
    setIsPlaying(false)
  }

  function handleSeek(time: number, isClip: boolean) {
    if (!syncPlay) return
    
    const cv = clipRef.current
    const mv = movieRef.current
    
    if (isClip && cv && mv) {
      // 从clip进度条拖拽，同步到movie
      const clipProgress = clipDuration > 0 ? time / clipDuration : 0
      const movieTime = clipProgress * movieDuration
      try {
        if (movieTime >= 0 && movieTime <= movieDuration) {
          mv.currentTime = movieTime
        }
      } catch {}
    } else if (!isClip && cv && mv) {
      // 从movie进度条拖拽，同步到clip
      const movieProgress = movieDuration > 0 ? time / movieDuration : 0
      const clipTime = movieProgress * clipDuration
      try {
        if (clipTime >= 0 && clipTime <= clipDuration) {
          cv.currentTime = clipTime
        }
      } catch {}
    }
  }

  // When selection or candidate selection changes, seek
  useEffect(()=>{ seekTo() },[selectedSegId, selectedCandIdx, followMovie, candList])

  // Accept selected candidate for selected row
  async function acceptSelected(){
    const r = selectedRow
    if (!r) return
    const cand = (candList && candList[selectedCandIdx]) || (r.top_matches && r.top_matches[selectedCandIdx]) || null
    if (!cand) return
    await applyChanges([{ seg_id: r.seg_id, chosen: cand }])
    await refreshOverrides()
    
    // 重新加载所有段落数据
    await refreshScenes()
    
    // 尝试选择下一个段落
    const currentIdx = allSegments.findIndex(x=>x.seg_id===r.seg_id)
    const nextIdx = currentIdx >= 0 && currentIdx + 1 < allSegments.length ? currentIdx + 1 : currentIdx
    if (nextIdx >= 0 && allSegments[nextIdx]) {
      setSelectedSegId(allSegments[nextIdx].seg_id)
    }
    setSelectedCandIdx(0)
  }

  // Scene level rebuild
  async function doRebuild(){
    if (activeScene==null) return
    await rebuildScene(activeScene)
    
    // 重新加载所有数据
    await refreshScenes()
    await refreshOverrides()
  }

  return <div className='layout'>
    <div className='toolbar'>
      <input style={{width:320}} placeholder='project root' value={root} onChange={e=>{ const v=e.target.value; setRoot(v); try{ localStorage.setItem('rm_root', v); }catch{} }} />
      <input style={{width:260}} placeholder='movie.mp4 (optional)' value={movie} onChange={e=>{ const v=e.target.value; setMovie(v); try{ localStorage.setItem('rm_movie', v); }catch{} }} />
      <input style={{width:260}} placeholder='clip.mp4 (optional)' value={clip} onChange={e=>{ const v=e.target.value; setClip(v); try{ localStorage.setItem('rm_clip', v); }catch{} }} />
      <button onClick={doOpen}>打开</button><button onClick={refreshScenes}>刷新场景</button><div style={{flex:1}}/>
      <label style={{marginRight:12}}><input type='checkbox' checked={followMovie} onChange={e=>setFollowMovie(e.target.checked)} /> 跟随Movie候选</label>
      <label style={{marginRight:12}}><input type='checkbox' checked={loop} onChange={e=>setLoop(e.target.checked)} /> 循环当前段</label>
      <span style={{marginRight:12, fontSize:12, opacity:0.8}}>
        循环次数: {loopCount}/{maxLoops}
        <input 
          type="number" 
          min="1" 
          max="10" 
          value={maxLoops} 
          onChange={e=>setMaxLoops(Math.max(1, parseInt(e.target.value) || 3))}
          style={{width:40, marginLeft:4, fontSize:11}}
        />
      </span>
      <label style={{marginRight:12}}><input type='checkbox' checked={mirrorClip} onChange={e=>setMirrorClip(e.target.checked)} /> 镜像Clip</label>
      <label style={{marginRight:12}}><input type='checkbox' checked={debug} onChange={e=>setDebug(e.target.checked)} /> 调试日志</label>
      <label style={{marginRight:12}}><input type='checkbox' checked={showDebugPanel} onChange={e=>setShowDebugPanel(e.target.checked)} /> 显示调试面板</label>
      <button onClick={refreshOverrides}>刷新覆盖</button>
      <button onClick={doRebuild}>场景内重建</button>
      <button onClick={()=>save()}>保存导出</button>
    </div>

    <div className='main'>
            <div className='panel'>
        <div style={{fontWeight:600,marginBottom:6}}>段落列表</div>
        <div className='scene-list' style={{maxHeight: '400px', overflowY: 'auto'}}>
          {Array.isArray(allSegments) && allSegments.length===0 && (
            <div style={{fontSize:12,opacity:.7,padding:'8px 4px'}}>无段落数据</div>
          )}
          {Array.isArray(allSegments) && allSegments.map((seg:SegmentRow)=>{
            const mo = seg.matched_orig_seg || {}
            const clipSceneId = seg.clip.scene_id
            const origSegId = mo.seg_id
            const hasOverride = overrides?.data && overrides.data[seg.seg_id]
            
            return (
              <div key={seg.seg_id}
                   className='candidate'
                   onClick={()=>setSelectedSegId(seg.seg_id)}
                   style={{
                     borderColor: selectedSegId===seg.seg_id ? '#409eff' : '#eee',
                     background: selectedSegId===seg.seg_id ? '#f5fbff' :  '#fff',
                     cursor: 'pointer',
                     marginBottom: 4
                   }}>
                <div style={{display:'flex', justifyContent:'space-between', marginBottom:4}}>
                  <div>
                    #{seg.seg_id} {clipSceneId}
                    {hasOverride && <span style={{color:'#409eff', marginLeft:8}}>✓ </span>}
                  </div>
                  <div style={{fontWeight:600, fontSize:12, opacity:0.7}}>
                    {mo ? `scene ${mo.scene_id} / idx ${mo.scene_seg_idx}` : '-'}
                  </div>
                </div>
                <div style={{fontSize:12, opacity:0.7}}>
                  clip: {formatTime(seg.clip?.start ?? 0)} - {formatTime(seg.clip?.end ?? 0)}
                </div>
                {mo && (
                  <div style={{fontSize:12, opacity:0.7}}>
                    movie: {formatTime(mo.start ?? 0)} - {formatTime(mo.end ?? 0)}
                  </div>
                )}
                {mo && (
                  <div style={{fontSize:12, opacity:0.6, marginTop:2}}>
                    seg_id: {origSegId}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      <div className='panel'>
        <div style={{marginBottom: 16}}>
          <div style={{display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8}}>
            <button onClick={isPlaying ? handlePause : handlePlay}>
              {isPlaying ? '⏸️ 暂停' : '▶️ 播放'}
            </button>
            <label style={{display: 'flex', alignItems: 'center', gap: 4}}>
              <input 
                type="checkbox" 
                checked={syncPlay} 
                onChange={(e) => setSyncPlay(e.target.checked)} 
              />
              同步播放
            </label>
          </div>
        </div>
        
        <div className='videos'>
          <div>
            <div style={{fontSize:12,opacity:.7,marginBottom:4}}>Clip</div>
            <video preload="metadata" ref={clipRef} src={clipSrc} controls={!syncPlay}
                   style={{ transform: mirrorClip ? 'scaleX(-1)' : undefined, transformOrigin: '50% 50%' }}
                   onLoadedMetadata={e=>{ 
                     const v=e.currentTarget as HTMLVideoElement
                     setClipDuration(v.duration)
                     if (debug) console.log('[Clip] loadedmetadata dur=', v.duration) 
                   }}
                   onTimeUpdate={e=>{ 
                     const v = e.currentTarget as HTMLVideoElement
                     setClipCurrentTime(v.currentTime)
                     const now=Date.now()
                     if (now - clipLastLogRef.current > 1000 && debug){ 
                       clipLastLogRef.current = now
                       console.log('[Clip] t=', v.currentTime) 
                     }
                   }}
                   onPlay={() => { 
                     if (!syncPlay) setIsPlaying(true)
                   }}
                   onPause={() => { 
                     if (!syncPlay) setIsPlaying(false)
                   }}
                   onSeeking={e=>{ 
                     clipSeekingRef.current = true
                     if (syncPlay && !movieSeekingRef.current) {
                       handleSeek(e.currentTarget.currentTime, true)
                     }
                     if (debug) console.log('[Clip] seeking to', e.currentTarget.currentTime) 
                   }}
                   onSeeked={e=>{ clipSeekingRef.current = false; if (debug) console.log('[Clip] seeked to', e.currentTarget.currentTime) }}
                   onError={e=>{ console.error('[Clip] error', e) }}
            />
            {syncPlay && selectedRow && (
              <div style={{marginTop: 4}}>
                <div style={{fontSize: 11, opacity: 0.7, marginBottom: 2}}>
                  {(() => {
                    // 优先使用当前播放的时间范围，如果没有则使用默认逻辑
                    if (currentPlayingTimeRange) {
                      const { clipStart, clipEnd } = currentPlayingTimeRange
                      return `${formatTime(clipStart + clipCurrentTime)} / ${formatTime(clipEnd)}`
                    } else {
                      // 降级到原有逻辑
                      return `${formatTime((selectedRow.clip?.start ?? 0) + clipCurrentTime)} / ${formatTime(selectedRow.clip?.end ?? (selectedRow.clip?.start ?? 0) + clipDuration)}`
                    }
                  })()}
                </div>
                <input
                  type="range"
                  min={0}
                  max={clipDuration || 1}
                  step={0.1}
                  value={clipCurrentTime}
                  onChange={(e) => {
                    const time = parseFloat(e.target.value)
                    if (clipRef.current) {
                      clipRef.current.currentTime = time
                      handleSeek(time, true)
                    }
                  }}
                  style={{width: '100%'}}
                />
              </div>
            )}
          </div>
          
          <div>
            <div style={{fontSize:12,opacity:.7,marginBottom:4}}>Movie</div>
            <video preload="metadata" ref={movieRef} src={movieSrc} controls={!syncPlay}
                   onLoadedMetadata={e=>{ 
                     const v=e.currentTarget as HTMLVideoElement
                     setMovieDuration(v.duration)
                     if (debug) console.log('[Movie] loadedmetadata dur=', v.duration) 
                   }}
                   onTimeUpdate={e=>{ 
                     const v = e.currentTarget as HTMLVideoElement
                     setMovieCurrentTime(v.currentTime)
                     const now=Date.now()
                     if (now - movieLastLogRef.current > 1000 && debug){ 
                       movieLastLogRef.current = now
                       console.log('[Movie] t=', v.currentTime) 
                     }
                   }}
                   onPlay={() => { 
                     if (!syncPlay) setIsPlaying(true)
                   }}
                   onPause={() => { 
                     if (!syncPlay) setIsPlaying(false)
                   }}
                   onSeeking={e=>{ 
                     movieSeekingRef.current = true
                     if (syncPlay && !clipSeekingRef.current) {
                       handleSeek(e.currentTarget.currentTime, false)
                     }
                     if (debug) console.log('[Movie] seeking to', e.currentTarget.currentTime) 
                   }}
                   onSeeked={e=>{ movieSeekingRef.current = false; if (debug) console.log('[Movie] seeked to', e.currentTarget.currentTime) }}
                   onError={e=>{ console.error('[Movie] error', e) }}
            />
            {syncPlay && selectedRow && (
              <div style={{marginTop: 4}}>
                <div style={{fontSize: 11, opacity: 0.7, marginBottom: 2}}>
                  {(() => {
                    // 优先使用当前播放的时间范围，如果没有则使用默认逻辑
                    if (currentPlayingTimeRange) {
                      const { movieStart, movieEnd } = currentPlayingTimeRange
                      return `${formatTime(movieStart + movieCurrentTime)} / ${formatTime(movieEnd)}`
                    } else {
                      // 降级到原有逻辑
                      const mo = selectedRow.matched_orig_seg || (candList && candList[selectedCandIdx])
                      const movieStart = mo?.start ?? 0
                      const movieEnd = mo?.end ?? (movieStart + movieDuration)
                      return `${formatTime(movieStart + movieCurrentTime)} / ${formatTime(movieEnd)}`
                    }
                  })()}
                </div>
                <input
                  type="range"
                  min={0}
                  max={movieDuration || 1}
                  step={0.1}
                  value={movieCurrentTime}
                  onChange={(e) => {
                    const time = parseFloat(e.target.value)
                    if (movieRef.current) {
                      movieRef.current.currentTime = time
                      handleSeek(time, false)
                    }
                  }}
                  style={{width: '100%'}}
                />
              </div>
            )}
          </div>
        </div>

        <table className='seg-table'>
          <thead><tr><th>seg_id</th><th>clip idx</th><th>clip t</th><th>orig t</th><th>matched scene/idx</th><th>score</th><th>操作</th></tr></thead>
          <tbody>
            {segments.map((s:SegmentRow)=>{
              const mo = s.matched_orig_seg || {}
              // 格式化 clip 时间
              const clipStart = s.clip.start ?? 0
              const clipEnd = s.clip.end ?? 0
              const clipTime = `${formatTime(clipStart)} - ${formatTime(clipEnd)}`
              
              // 格式化 orig 时间
              const origStart = mo.start ?? 0
              const origEnd = mo.end ?? 0
              const origTime = mo.start !== undefined ? `${formatTime(origStart)} - ${formatTime(origEnd)}` : '-'
              
              const isSel = selectedSegId===s.seg_id
              const isPlaying = playingSegId === s.seg_id && playingFromSide === 'left'
              
              let bgColor = 'transparent'
              if (isPlaying) {
                bgColor = '#fff5f2' // 橙色背景表示正在播放
              } else if (isSel) {
                bgColor = '#f7fbff' // 蓝色背景表示选中
              }
              
              return <tr key={s.seg_id}
                         style={{
                           background: bgColor, 
                           cursor:'pointer',
                           borderLeft: isPlaying ? '4px solid #ff6b35' : isSel ? '4px solid #409eff' : '4px solid transparent'
                         }}
                         onClick={()=>{ 
                           setSelectedSegId(s.seg_id); 
                           setSelectedCandIdx(0); 
                           // 更新当前校对的clip segment信息
                           setCurrentClipSegment({
                             segId: s.seg_id,
                             clipStart: s.clip.start ?? 0,
                             clipEnd: s.clip.end ?? 0,
                             sceneId: s.clip.scene_id,
                             sceneSegIdx: s.clip.scene_seg_idx
                           });
                           seekTo(s,0);
                         } }>
                <td style={{ fontWeight: isPlaying ? 'bold' : 'normal', color: isPlaying ? '#ff6b35' : 'inherit' }}>
                  {isPlaying && <span style={{ marginRight: 4 }}>▶</span>}
                  {s.seg_id}
                </td>
                <td>{s.clip.scene_seg_idx}</td>
                <td style={{ fontSize: '12px' }}>{clipTime}</td>
                <td style={{ fontSize: '12px' }}>{origTime}</td>
                <td>{mo.scene_id??'-'} / {mo.scene_seg_idx??'-'}</td>
                <td>{(mo.score??0).toFixed(3)}</td>
                <td>
                  <button onClick={(e)=>{ e.stopPropagation(); acceptSelected() }}>接受候选</button>
                </td>
              </tr>
            })}
          </tbody>
        </table>
      </div>

      <div className='panel'>
        <div style={{display:'flex',alignItems:'center',marginBottom:6}}>
          <div style={{fontWeight:600}}>候选（当前段）</div>
          <div style={{marginLeft:12, display:'flex', gap:8}}>
            {(['top','scene','all'] as const).map(md=>(
              <button key={md}
                      onClick={()=>setCandMode(md)}
                      style={{padding:'4px 8px', border:'1px solid #ddd', borderRadius:4, background: candMode===md?'#eef6ff':'#fff'}}>
                {md==='top'?'Top': md==='scene'?'场景内':'全部'}
              </button>
            ))}
          </div>
          <div style={{marginLeft:'auto', fontSize:12, opacity:.7}}>共 {candList?.length ?? 0} 条（展示前 50）</div>
        </div>
        {!selectedRow && <div style={{fontSize:12,opacity:.7}}>选中一行以查看候选</div>}
        {selectedRow && <div>
          {showOrigSegments && candMode === 'scene' ? (
            // 显示原始段落列表 (按场景分组)
            <div>
              <div style={{fontSize:12, opacity:.7, marginBottom:8}}>
                场景 {selectedRow.matched_orig_seg?.scene_id} 及前后场景的段落 (共 {origSegments.length} 个)
              </div>
              <div style={{
                maxHeight: '400px',
                overflowY: 'auto',
                border: '1px solid #eee',
                borderRadius: '4px',
                padding: '8px',
                marginBottom: '12px'
              }}>
                {(() => {
                  // 按场景ID分组
                  const sceneGroups = new Map()
                  origSegments.forEach(seg => {
                    const sceneId = seg._sceneId
                    if (!sceneGroups.has(sceneId)) {
                      sceneGroups.set(sceneId, [])
                    }
                    sceneGroups.get(sceneId).push(seg)
                  })
                  
                  return Array.from(sceneGroups.entries()).map(([sceneId, segments]) => (
                    <div key={sceneId} style={{marginBottom: 16}}>
                      <div style={{
                        fontSize: 13, 
                        fontWeight: 600, 
                        marginBottom: 8,
                        color: segments[0]?._isCurrentScene ? '#409eff' : '#666',
                        borderBottom: '1px solid #eee',
                        paddingBottom: 4
                      }}>
                        场景 {sceneId} {segments[0]?._isCurrentScene ? '(当前)' : ''} - {segments.length} 个段落
                      </div>
                      {segments.map((origSeg: any, i: number) => {
                        // 检查这个原始段落是否在当前候选列表中
                        const isCandidate = candList.some(c => 
                          c.seg_id === origSeg.seg_id && c.scene_id === origSeg.scene_id
                        )
                        const isSelected = candList[selectedCandIdx]?.seg_id === origSeg.seg_id
                        const isPlaying = playingType === 'orig' && playingOrigSegId === origSeg.seg_id
                        
                        // 颜色优先级：播放中 > 选中 > 候选 > 默认
                        let borderColor = '#eee'
                        let backgroundColor = '#fff'
                        if (isPlaying) {
                          borderColor = '#ff6b35'
                          backgroundColor = '#fff5f2'
                        } else if (isSelected) {
                          borderColor = '#409eff'
                          backgroundColor = '#f5fbff'
                        } else if (isCandidate) {
                          borderColor = '#67c23a'
                          backgroundColor = '#f0f9ff'
                        }
                        
                        return (
                          <div key={`${sceneId}-${i}`} 
                               className='candidate'
                               style={{
                                 borderColor,
                                 background: backgroundColor,
                                 opacity: origSeg._isCurrentScene ? (isCandidate ? 1 : 0.8) : (isCandidate ? 1 : 0.4),
                                 cursor: 'pointer',
                                 marginBottom: 4,
                                 boxShadow: isPlaying ? '0 0 8px rgba(255, 107, 53, 0.3)' : undefined
                               }}
                               onClick={() => {
                                 if (isCandidate) {
                                   // 对于候选项，使用 seekToCandidate 逻辑
                                   const candIdx = candList.findIndex(c => 
                                     c.seg_id === origSeg.seg_id && c.scene_id === origSeg.scene_id
                                   )
                                   if (candIdx >= 0) {
                                     seekToCandidate(candList[candIdx], candIdx)
                                   } else {
                                     // 如果在candList中找不到，使用备用方案
                                     seekToOrigSegment(origSeg)
                                   }
                                 } else {
                                   // 对于非候选项，使用独立的播放函数
                                   seekToOrigSegment(origSeg)
                                 }
                               }}>
                            <div style={{display:'flex',justifyContent:'space-between',marginBottom:4}}>
                              <div style={{ 
                                fontWeight: isPlaying ? 'bold' : 'normal',
                                color: isPlaying ? '#ff6b35' : '#333'
                              }}>
                                {isPlaying && <span style={{ marginRight: 4, color: '#ff6b35' }}>▶</span>}
                                seg {origSeg.seg_id} / idx {origSeg.scene_seg_idx}
                                {isCandidate && <span style={{color:'#67c23a', marginLeft:8}}>✓ 候选</span>}
                                {!origSeg._isCurrentScene && <span style={{color:'#999', marginLeft:8, fontSize:11}}>其他场景</span>}
                              </div>
                              <div style={{fontWeight:600}}>
                                {isCandidate ? 
                                  (candList.find(c => c.seg_id === origSeg.seg_id)?.score?.toFixed?.(3) ?? '-') 
                                  : '-'
                                }
                              </div>
                            </div>
                            <div style={{
                              fontSize:12,
                              opacity: isPlaying ? 1 : .7,
                              fontWeight: isPlaying ? 'bold' : 'normal',
                              color: isPlaying ? '#ff6b35' : 'inherit'
                            }}>
                              {formatTime(origSeg.start ?? 0)} - {formatTime(origSeg.end ?? 0)}
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  ))
                })()}
              </div>
            </div>
          ) : (
            // 原有的候选项列表
            (candList||[]).slice(0,50).map((c:any,i:number)=>{
              const isSelected = i===selectedCandIdx
              // 修正播放状态检测：简化逻辑，只检查选中状态和播放类型
              const isPlaying = playingType === 'clip' && 
                               playingFromSide === 'right' && 
                               isSelected
              
              // 颜色优先级：播放中 > 选中 > 默认
              let borderColor = '#eee'
              let backgroundColor = '#fff'
              if (isPlaying) {
                borderColor = '#ff6b35'
                backgroundColor = '#fff5f2'
              } else if (isSelected) {
                borderColor = '#409eff'
                backgroundColor = '#f5fbff'
              }
              
              return <div key={i} className='candidate'
                          style={{
                            borderColor, 
                            background: backgroundColor,
                            boxShadow: isPlaying ? '0 0 8px rgba(255, 107, 53, 0.3)' : undefined
                          }}
                          onClick={()=>{ seekToCandidate(c, i) }}>
                <div style={{display:'flex',justifyContent:'space-between',marginBottom:4}}>
                  <div style={{ 
                    fontWeight: isPlaying ? 'bold' : 'normal',
                    color: isPlaying ? '#ff6b35' : '#333'
                  }}>
                    {isPlaying && <span style={{ marginRight: 4 }}>▶</span>}
                    scene {c.scene_id} / idx {c.scene_seg_idx}
                  </div>
                  <div style={{fontWeight:600}}>{(c.score??0).toFixed?.(3) ?? c.score}</div>
                </div>
                <div style={{
                  fontSize:12,
                  opacity: isPlaying ? 1 : .7,
                  fontWeight: isPlaying ? 'bold' : 'normal',
                  color: isPlaying ? '#ff6b35' : 'inherit'
                }}>
                  {formatTime(c.start ?? 0)} - {formatTime(c.end ?? 0)}
                </div>
                <div style={{fontSize:12,opacity:.6, marginTop:2}}>src: {c.source || '-'}</div>
              </div>
            })
          )}
          <div style={{display:'flex', gap:8}}>
            <button onClick={acceptSelected}>应用所选</button>
            <button onClick={()=>{ 
              if (candMode === 'scene' && origSegments.length > 0) {
                // 在场景模式下，播放第一个原始段落
                const firstOrigSeg = origSegments[0]
                if (firstOrigSeg) {
                  setSelectedCandIdx(0)
                  seekToOrigSegment(firstOrigSeg)
                }
              } else if (candList && candList.length > 0) {
                // 在其他模式下，播放第一个候选项
                seekToCandidate(candList[0], 0)
              }
            }}>选第一个</button>
          </div>
        </div>}
      </div>

      {showDebugPanel && (
        <div className='panel'>
          <div style={{display:'flex', alignItems:'center', marginBottom:6}}>
            <div style={{fontWeight:600}}>调试</div>
            <div style={{marginLeft:12, fontSize:12, opacity:.7}}>sidecar: {overrides?.path || '-'}</div>
            <div style={{marginLeft:12, fontSize:12, opacity:.7}}>count: {overrides?.count ?? '-'}</div>
            <div style={{flex:1}}/>
            <button onClick={clearOverrides}>清空覆盖</button>
          </div>
          <div style={{display:'grid', gridTemplateColumns:'1fr 2fr', gap:12}}>
            <div>
              <div style={{fontSize:12, opacity:.7, marginBottom:4}}>覆盖列表（前 50 条）</div>
              <div style={{maxHeight:240, overflow:'auto', border:'1px solid #eee', borderRadius:4, padding:8}}>
                {overrides?.data ?
                  Object.entries(overrides.data).slice(0,50).map(([k,v]: any) => (
                    <div key={k} style={{padding:'4px 0', borderBottom:'1px dashed #eee'}}>
                      <div style={{fontWeight:600}}>seg {k}</div>
                      <div style={{fontSize:12, opacity:.8}}>scene {v?.scene_id} / idx {v?.scene_seg_idx} | score {(v?.score??0).toFixed?.(3)}</div>
                    </div>
                  ))
                  : <div style={{fontSize:12, opacity:.6}}>暂无数据</div>}
              </div>
            </div>
            <div>
              <div style={{fontSize:12, opacity:.7, marginBottom:4}}>原始 JSON（截断显示）</div>
              <div style={{maxHeight:240, overflow:'auto', border:'1px solid #eee', borderRadius:4, padding:8}}>
                <pre style={{margin:0, fontSize:12}}>{JSON.stringify(overrides?.data ?? {}, null, 2).slice(0, 4000)}</pre>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  </div>
}