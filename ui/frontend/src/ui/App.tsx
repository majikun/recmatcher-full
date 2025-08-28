import React, {useEffect, useMemo, useRef, useState} from 'react'
import {openProject,listScenes,listSegments,applyChanges,save,rebuildScene, listOrigSegments} from '../api'
import { formatTime, usePersistedState, useCandidates, usePlayerController, type Scene, type SegmentRow, type Candidate } from './logic'

const BACKEND_BASE = `${window.location.protocol}//${window.location.hostname}:8787`

export default function App(){
  const [root, setRoot]   = usePersistedState<string>('rm_root', '')
  const [movie, setMovie] = usePersistedState<string>('rm_movie', '')
  const [clip, setClip]   = usePersistedState<string>('rm_clip', '')

  const [scenes,setScenes]=useState<Scene[]>([])
  const [activeScene,setActiveScene]=useState<number|null>(null)
  const [segments,setSegments]=useState<SegmentRow[]>([])
  const [allSegments,setAllSegments]=useState<SegmentRow[]>([])
  const [selectedSegId,setSelectedSegId]=useState<number|null>(null)
  const [selectedCandIdx,setSelectedCandIdx]=useState<number>(0)
  // 当前准备应用的候选（可能来自右侧候选列表，也可能来自“场景内原片段/走廊”面板）
  const [pendingChoice, setPendingChoice] = useState<{type:'cand'|'orig', data:any} | null>(null)
  // 当前选中段对应的“选中原片段”（用于避免候选列表异步刷新把 Movie 的时间顶掉）
  const [selectedMo, setSelectedMo] = useState<any|null>(null)
  const [followMovie,setFollowMovie]=useState<boolean>(true)
  const [loop,setLoop]=useState<boolean>(true)
  const [mirrorClip, setMirrorClip] = useState<boolean>(false)
  const [candMode, setCandMode] = useState<'top'|'scene'|'all'|'corridor'>('top')
  const { items: candList, total: candTotal } = useCandidates(selectedSegId, candMode, 120)
  const [origSegments, setOrigSegments] = useState<any[]>([])
  const [showOrigSegments, setShowOrigSegments] = useState<boolean>(false)

  // 走廊（前/后 clip 场景锚点相邻原片场景）
  const [corridorPrev, setCorridorPrev] = useState<any[]>([])
  const [corridorNext, setCorridorNext] = useState<any[]>([])
  const corridorCacheRef = useRef(new Map<number, any[]>())
  const CORRIDOR_N = 2 // 默认展示相邻 2 个场景

  const clipRef = useRef<HTMLVideoElement|null>(null)
  const movieRef = useRef<HTMLVideoElement|null>(null)

  // player refs above

  const [debug, setDebug] = useState<boolean>(true)
  const clipLastLogRef = useRef<number>(0)
  const movieLastLogRef = useRef<number>(0)
  const dlog = (...args:any[]) => { if (debug) console.log('[UI]', ...args) }

  const [showDebugPanel, setShowDebugPanel] = useState<boolean>(true)
  const [overrides, setOverrides] = useState<any>(null)
  const [sceneHints, setSceneHints] = useState<Record<number, {scene_id:number, scene_seg_idx:number} | null>>({})
    // 兼容 overrides 的 key 既可能是数字也可能是字符串
  const getOverrideForSeg = (segId: number) => {
    const d: any = overrides?.data || {}
    return d[segId] ?? d[String(segId)] ?? null
  }

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
  // unified player controller
  const [maxLoops, setMaxLoops] = useState<number>(3)
  const { isPlaying, loopCount, range, playPair, play: playSync, pause: pauseSync, seekClipRel, seekMovieRel } =
    usePlayerController({
      clipRef,
      movieRef,
      backendBase: BACKEND_BASE,
      syncPlay,
      maxLoops,
      debug,
      onSetSrc: (cUrl, mUrl) => { setClipSrc(cUrl); setMovieSrc(mUrl) }
    })
  
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

  // 最近一次 seek 的来源：用于避免左侧点击后被自动 useEffect 立即覆盖
  const lastSeekByRef = useRef<'left'|'right'|'auto'|null>(null)
  

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
    await Promise.all([
      refreshOverrides(),
      refreshScenes(),
    ])
  }

  // load segments when selection changes
  useEffect(()=>{
    if(selectedSegId != null){
      // 当选择了 seg_id 时，从 allSegments 中找到对应的段落
      const selectedSeg = allSegments.find(s => s.seg_id === selectedSegId)
      if (selectedSeg) {
        // 设置当前活动场景为该段落所属的场景
        setActiveScene(selectedSeg.clip.scene_id || null)
        // 设置默认的“当前原片段”为服务端已应用的匹配
        const merged = (selectedSeg as any).matched_orig_seg || null
        setSelectedMo(merged)
        // 加载该场景的所有段落
        if (selectedSeg.clip.scene_id != null) {
          listSegments(selectedSeg.clip.scene_id).then(arr=>{
            setSegments(arr)
            // remove loadCandidates
          }).catch(e => {
            console.error('Failed to load segments', e)
          })
        }
      }
    }
  },[selectedSegId, allSegments])

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

  // 加载走廊：以锚点 scene_id 为基准，prev=向后取、next=向前取相邻原片场景
  async function loadCorridorFor(anchorSceneId: number, dir: 'prev'|'next', n = CORRIDOR_N){
    const ids: number[] = []
    if (dir === 'prev'){
      for (let s = anchorSceneId + 1; s <= anchorSceneId + n; s++) ids.push(s)
    } else {
      for (let s = anchorSceneId - n; s <= anchorSceneId - 1; s++) if (s > 0) ids.push(s)
    }
    const all: any[] = []
    for (const sid of ids){
      if (!corridorCacheRef.current.has(sid)){
        try{
          const resp = await listOrigSegments(sid)
          const arr = (resp?.segments || []).map((x:any)=>({ ...x, _sceneId: sid, _corridor: dir }))
          corridorCacheRef.current.set(sid, arr)
        }catch(e){ console.error('loadCorridorFor failed', e); corridorCacheRef.current.set(sid, []) }
      }
      all.push(...(corridorCacheRef.current.get(sid) || []))
    }
    if (dir==='prev') setCorridorPrev(all); else setCorridorNext(all)
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

  // 当选择变化时，基于前/后 clip 场景的原片锚点，加载走廊
  useEffect(()=>{
    if (!selectedRow) return
    const cid = selectedRow.clip?.scene_id
    if (cid == null) return

    // 取所有 clip 场景 id，去重排序
    const sceneIds = Array.from(new Set(allSegments.map(s=>s.clip?.scene_id).filter((x:any)=>x!=null))).sort((a:any,b:any)=>a-b)
    const idx = sceneIds.indexOf(cid)
    const prevCid = idx>0 ? sceneIds[idx-1] : null
    const nextCid = (idx>=0 && idx<sceneIds.length-1) ? sceneIds[idx+1] : null

    // 从 sceneHints 或首行匹配里取锚点原片 scene_id
    const anchorFromClipScene = (clipSceneId:number|null)=>{
      if (!clipSceneId) return null
      const hint = sceneHints[clipSceneId]
      if (hint?.scene_id != null) return hint.scene_id
      const firstRow = allSegments.find(s=>s.clip?.scene_id===clipSceneId)
      const mo:any = firstRow?.matched_orig_seg || firstRow?.top_matches?.[0]
      return mo?.scene_id ?? null
    }

    const anchorPrev = anchorFromClipScene(prevCid)
    const anchorNext = anchorFromClipScene(nextCid)

    setCorridorPrev([]); setCorridorNext([])
    if (anchorPrev) loadCorridorFor(anchorPrev, 'prev', CORRIDOR_N)
    if (anchorNext) loadCorridorFor(anchorNext, 'next', CORRIDOR_N)
  }, [selectedRow, allSegments, sceneHints])

  // Seek videos to the currently selected row (clip uses clip times, movie uses current choice/candidate)
  function seekTo(row?: SegmentRow, candIdx?: number){
    const r = row || selectedRow
    if (!r) return
    const clipStart = r.clip.start ?? 0
    const clipEnd   = r.clip.end   ?? (clipStart + 2)
    let mo: any = selectedMo || (r as any).matched_orig_seg || {}
    const cand = (candList && candList[candIdx ?? selectedCandIdx]) || ((r.top_matches && r.top_matches[candIdx ?? selectedCandIdx]) || null)
    const hasOv = !!(r as any).is_override
    // 只有在“没有覆盖”和“没有手动选择 selectedMo”的情况下才允许跟随候选
    if (!selectedMo && followMovie && !hasOv && cand) mo = cand
    const movStart = mo?.start ?? 0
    const movEnd   = mo?.end   ?? (movStart + 2)
    playPair(clipStart, clipEnd, movStart, movEnd)
    // 更新播放态用于高亮
    setPlayingSegId(r.seg_id)
    setPlayingFromSide('left')
    setPlayingType('clip')
    setPlayingOrigSegId(mo?.seg_id ?? null)
    // 维护当前校对段信息（供右侧使用）
    setCurrentClipSegment({ segId: r.seg_id, clipStart, clipEnd, sceneId: r.clip.scene_id, sceneSegIdx: r.clip.scene_seg_idx })
  }

  // 左侧列表点击播放：严格使用服务端匹配；播放中点击直接忽略
  function seekFromLeft(seg: SegmentRow){
    if (isPlaying){ dlog('left click ignored: playing'); return }
    const clipStart = seg.clip?.start ?? 0
    const clipEnd   = seg.clip?.end   ?? (clipStart + 2)
    const mo: any = (seg as any).matched_orig_seg || {}
    setSelectedMo(mo)
    const movStart = mo?.start ?? 0
    const movEnd   = mo?.end   ?? (movStart + 2)
    lastSeekByRef.current = 'left'
    // 使用统一控制器
    playPair(clipStart, clipEnd, movStart, movEnd)
    // 选中并高亮播放态
    setSelectedSegId(seg.seg_id)
    setPlayingSegId(seg.seg_id)
    setPlayingFromSide('left')
    setPlayingType('clip')
    setPlayingOrigSegId(mo?.seg_id ?? null)
    setPendingChoice(null)
    setCurrentClipSegment({
      segId: seg.seg_id,
      clipStart,
      clipEnd,
      sceneId: seg.clip.scene_id,
      sceneSegIdx: seg.clip.scene_seg_idx
    })
  }

  function seekToOrigSegment(origSeg: any) {
    if (!origSeg || !currentClipSegment) return
    const clipStart = currentClipSegment.clipStart
    const clipEnd   = currentClipSegment.clipEnd
    const movStart  = origSeg.start ?? 0
    const movEnd    = origSeg.end   ?? (movStart + 2)
    playPair(clipStart, clipEnd, movStart, movEnd)
    setPlayingSegId(currentClipSegment.segId)
    setPlayingFromSide('right')
    setPlayingType('orig')
    setPlayingOrigSegId(origSeg.seg_id ?? null)
    // 将原片段映射为 candidate-like 数据，便于 apply 使用
    const mapped = {
      seg_id: origSeg.seg_id,
      scene_seg_idx: origSeg.scene_seg_idx,
      start: origSeg.start,
      end: origSeg.end,
      scene_id: origSeg.scene_id,
      score: origSeg.score ?? 0,
      faiss_id: origSeg.faiss_id ?? undefined,
      movie_id: 'movie',
      shot_id: -1,
      source: origSeg.source ?? 'scene'
    }
    setPendingChoice({ type: 'orig', data: mapped })
    setSelectedMo(mapped || null)
  }

  function seekToCandidate(candidate: any, candIdx: number) {
    if (!currentClipSegment || !candidate) return
    const clipStart = currentClipSegment.clipStart
    const clipEnd   = currentClipSegment.clipEnd
    const movStart  = candidate.start ?? 0
    const movEnd    = candidate.end   ?? (movStart + 2)
    setSelectedCandIdx(candIdx)
    setPendingChoice({ type: 'cand', data: candidate })
    setSelectedMo(candidate || null)
    lastSeekByRef.current = 'right'
    playPair(clipStart, clipEnd, movStart, movEnd)
    setPlayingSegId(currentClipSegment.segId)
    setPlayingFromSide('right')
    setPlayingType('clip')
    setPlayingOrigSegId(candidate.seg_id ?? null)
  }

  // When selection or candidate selection changes, seek
  useEffect(()=>{
    if (lastSeekByRef.current === 'left') { // 左侧点击刚刚触发过精确 seek，避免被自动 seek 覆盖
      lastSeekByRef.current = null
      return
    }
    seekTo()
  },[selectedSegId, selectedCandIdx, followMovie, candList])

  // Accept selected candidate for selected row
  async function acceptSelected(){
    const r = selectedRow
    if (!r) { console.warn('[apply] no selected row'); return }

    // 优先使用 pendingChoice（可能来自场景内原片段/走廊）；其次回退到当前候选列表；再回退 row.top_matches
    const fromPending = pendingChoice?.data
    const fromCandList = (candList && candList[selectedCandIdx]) || null
    const fromRowTop = (r.top_matches && r.top_matches[selectedCandIdx]) || null
    const chosen = fromPending || fromCandList || fromRowTop

    if (!chosen){ console.warn('[apply] no candidate to apply'); return }

    const change = { seg_id: r.seg_id, chosen }

    try{
      console.log('[apply] sending change', change)
      await applyChanges([change])

      // 本地乐观更新：更新中间表（segments）里该行
      setSegments(prev => Array.isArray(prev) ? prev.map(row => row.seg_id===r.seg_id ? ({...row, matched_orig_seg: {...chosen}, is_override: true, matched_source: 'applied'}) : row) : prev)
      // 也更新左侧 allSegments（段落列表）以保持一致
      setAllSegments(prev => Array.isArray(prev) ? prev.map(row => row.seg_id===r.seg_id ? ({...row, matched_orig_seg: {...chosen}, is_override: true, matched_source: 'applied'}) : row) : prev)

      // 乐观更新本地 overrides，立即点亮左侧 ✓ 标记（保留以便调试面板查看）
      setOverrides(prev => {
        const data = { ...(prev?.data || {}), [String(r.seg_id)]: { ...chosen } }
        return { ...(prev || {}), data, count: Object.keys(data).length }
      })

      console.log('[apply] applied on seg', r.seg_id, '->', chosen)
      await refreshOverrides()
    } catch (e:any) {
      console.error('[apply] failed', e)
      alert('应用失败: ' + (e?.message || String(e)))
    }
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
      <input style={{width:320}} placeholder='project root' value={root} onChange={e=>{ setRoot(e.target.value) }} />
      <input style={{width:260}} placeholder='movie.mp4 (optional)' value={movie} onChange={e=>{ setMovie(e.target.value) }} />
      <input style={{width:260}} placeholder='clip.mp4 (optional)' value={clip} onChange={e=>{ setClip(e.target.value) }} />
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
            const mo: any = seg.matched_orig_seg || null
            const clipSceneId = seg.clip.scene_id
            const origSegId = mo?.seg_id
            const hasOverride = !!(seg as any).is_override

            return (
              <div key={seg.seg_id}
                   className='candidate'
                   onClick={()=>{ setSelectedSegId(seg.seg_id); seekFromLeft(seg) }}
                   style={{
                     borderColor: selectedSegId===seg.seg_id ? '#409eff' : '#eee',
                     background: selectedSegId===seg.seg_id ? '#f5fbff' :  '#fff',
                     cursor: 'pointer',
                     marginBottom: 4
                   }}>
                <div style={{display:'flex', justifyContent:'space-between', marginBottom:4}}>
                  <div>
                    #{seg.seg_id} S{clipSceneId}/idx {seg.clip.scene_seg_idx}
                    {hasOverride && <span style={{color:'#409eff', marginLeft:8}}>✓ </span>}
                  </div>
                  <div style={{fontWeight:600, fontSize:12, opacity:0.7}}>
                    seg {origSegId} {mo ? `S${mo.scene_id} / idx ${mo.scene_seg_idx}` : '-'}
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
                    {mo ? `scene ${mo.scene_id} / idx ${mo.scene_seg_idx}` : '-'}
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
            <button onClick={isPlaying ? pauseSync : playSync}>
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
                     //if (!syncPlay) setIsPlaying(true)
                   }}
                   onPause={() => { 
                     //if (!syncPlay) setIsPlaying(false)
                   }}
                   onSeeking={e=>{ 
                     if (debug) console.log('[Clip] seeking to', e.currentTarget.currentTime) 
                   }}
                   onSeeked={e=>{ if (debug) console.log('[Clip] seeked to', e.currentTarget.currentTime) }}
                   onError={e=>{ console.error('[Clip] error', e) }}
            />
            {syncPlay && selectedRow && (
              <div style={{marginTop: 4}}>
                <div style={{fontSize: 11, opacity: 0.7, marginBottom: 2}}>
                  { range ? `${formatTime(range.clipStart + clipCurrentTime)} / ${formatTime(range.clipEnd)}`
                          : `${formatTime((selectedRow?.clip?.start ?? 0) + clipCurrentTime)} / ${formatTime(selectedRow?.clip?.end ?? (selectedRow?.clip?.start ?? 0) + clipDuration)}` }
                </div>
                <input
                  type="range"
                  min={0}
                  max={clipDuration || 1}
                  step={0.1}
                  value={clipCurrentTime}
                  onChange={(e) => {
                    const t = parseFloat(e.target.value)
                    if (clipRef.current) clipRef.current.currentTime = t
                    seekClipRel(t)
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
                     //if (!syncPlay) setIsPlaying(true)
                   }}
                   onPause={() => { 
                     //if (!syncPlay) setIsPlaying(false)
                   }}
                   onSeeking={e=>{ 
                     if (debug) console.log('[Movie] seeking to', e.currentTarget.currentTime) 
                   }}
                   onSeeked={e=>{ if (debug) console.log('[Movie] seeked to', e.currentTarget.currentTime) }}
                   onError={e=>{ console.error('[Movie] error', e) }}
            />
            {syncPlay && selectedRow && (
              <div style={{marginTop: 4}}>
                <div style={{fontSize: 11, opacity: 0.7, marginBottom: 2}}>
                  { range ? `${formatTime(range.movieStart + movieCurrentTime)} / ${formatTime(range.movieEnd)}` 
                  : (()=>{ 
                      if (!selectedRow) return `${formatTime(0)} / ${formatTime(movieDuration)}`
                      const hasOv = !!(selectedRow as any).is_override
                      let mo:any = selectedMo || (selectedRow as any).matched_orig_seg || {}
                      if (!selectedMo && followMovie && !hasOv && candList) {
                        const c = candList[selectedCandIdx]
                        if (c) mo = c
                      }
                      const ms=mo?.start??0; const me=mo?.end??(ms+movieDuration)
                      return `${formatTime(ms + movieCurrentTime)} / ${formatTime(me)}`
                    })() }
                </div>
                <input
                  type="range"
                  min={0}
                  max={movieDuration || 1}
                  step={0.1}
                  value={movieCurrentTime}
                  onChange={(e) => {
                    const t = parseFloat(e.target.value)
                    if (movieRef.current) movieRef.current.currentTime = t
                    seekMovieRel(t)
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
              const mo: any = (s as any).matched_orig_seg || {}
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
                           setPendingChoice(null);
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
                <td>seg{mo.seg_id??'-'} S{mo.scene_id??'-'} / idx {mo.scene_seg_idx??'-'}</td>
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
            {(['top','scene','corridor','all'] as const).map(md=>(
              <button key={md}
                      onClick={()=>setCandMode(md)}
                      style={{padding:'4px 8px', border:'1px solid #ddd', borderRadius:4, background: candMode===md?'#eef6ff':'#fff'}}>
                {md==='top'?'Top': md==='scene'?'场景内': md==='corridor'?'走廊':'全部'}
              </button>
            ))}
          </div>
          <div style={{marginLeft:'auto', fontSize:12, opacity:.7}}>共 {candTotal ?? (candList?.length ?? 0)} 条（展示前 50）</div>
        </div>
        {!selectedRow && <div style={{fontSize:12,opacity:.7}}>选中一行以查看候选</div>}
        {selectedRow && <div>
          {showOrigSegments && candMode === 'scene' ? (
            // === 场景内 ===
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
                                seg {origSeg.seg_id} S{origSeg._sceneId} / idx {origSeg.scene_seg_idx}
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
          ) : candMode === 'corridor' ? (
            // === 走廊 ===
            <div>
              <div style={{fontSize:12, opacity:.7, marginBottom:8}}>
                基于 clip 前后场景锚点，展示相邻原片场景（±{CORRIDOR_N}）
              </div>

              {/* 前序走廊 */}
              <div style={{fontSize:13, fontWeight:600, margin:'12px 0 8px'}}>← 前序走廊</div>
              <div style={{maxHeight: 200, overflowY: 'auto', border: '1px solid #eee', borderRadius: 4, padding: 8, marginBottom: 12}}>
                {corridorPrev.length===0 && <div style={{fontSize:12, opacity:.6}}>无数据</div>}
                {corridorPrev.map((origSeg:any, i:number)=>{
                  const isCandidate = candList.some(c => c.seg_id===origSeg.seg_id && c.scene_id===origSeg.scene_id)
                  const isSelected = candList[selectedCandIdx]?.seg_id === origSeg.seg_id
                  const isPlaying = playingType === 'orig' && playingOrigSegId === origSeg.seg_id
                  let borderColor = '#eee', backgroundColor = '#fff'
                  if (isPlaying){ borderColor = '#ff6b35'; backgroundColor = '#fff5f2' }
                  else if (isSelected){ borderColor = '#409eff'; backgroundColor = '#f5fbff' }
                  else if (isCandidate){ borderColor = '#67c23a'; backgroundColor = '#f0f9ff' }
                  return (
                    <div key={`prev-${i}`} className='candidate' style={{borderColor, background: backgroundColor, marginBottom:4, cursor:'pointer'}}
                         onClick={()=>{
                           if (isCandidate){
                             const candIdx = candList.findIndex(c => c.seg_id===origSeg.seg_id && c.scene_id===origSeg.scene_id)
                             if (candIdx>=0) seekToCandidate(candList[candIdx], candIdx); else seekToOrigSegment(origSeg)
                           }else{ seekToOrigSegment(origSeg) }
                         }}>
                      <div style={{display:'flex',justifyContent:'space-between',marginBottom:4}}>
                        <div style={{ fontWeight: isPlaying ? 'bold' : 'normal', color: isPlaying ? '#ff6b35' : '#333' }}>
                          {isPlaying && <span style={{ marginRight: 4 }}>▶</span>}
                          seg {origSeg.seg_id} S{origSeg._sceneId} / idx {origSeg.scene_seg_idx}
                          {isCandidate && <span style={{color:'#67c23a', marginLeft:8}}>✓ 候选</span>}
                        </div>
                        <div style={{fontWeight:600}}>{isCandidate ? (candList.find(c => c.seg_id===origSeg.seg_id)?.score?.toFixed?.(3) ?? '-') : '-'}</div>
                      </div>
                      <div style={{fontSize:12, opacity: isPlaying?1:.7, fontWeight: isPlaying?'bold':'normal', color: isPlaying?'#ff6b35':'inherit'}}>
                        {formatTime(origSeg.start ?? 0)} - {formatTime(origSeg.end ?? 0)}
                      </div>
                    </div>
                  )
                })}
              </div>

              {/* 后续走廊 */}
              <div style={{fontSize:13, fontWeight:600, margin:'12px 0 8px'}}>后续走廊 →</div>
              <div style={{maxHeight: 200, overflowY: 'auto', border: '1px solid #eee', borderRadius: 4, padding: 8}}>
                {corridorNext.length===0 && <div style={{fontSize:12, opacity:.6}}>无数据</div>}
                {corridorNext.map((origSeg:any, i:number)=>{
                  const isCandidate = candList.some(c => c.seg_id===origSeg.seg_id && c.scene_id===origSeg.scene_id)
                  const isSelected = candList[selectedCandIdx]?.seg_id === origSeg.seg_id
                  const isPlaying = playingType === 'orig' && playingOrigSegId === origSeg.seg_id
                  let borderColor = '#eee', backgroundColor = '#fff'
                  if (isPlaying){ borderColor = '#ff6b35'; backgroundColor = '#fff5f2' }
                  else if (isSelected){ borderColor = '#409eff'; backgroundColor = '#f5fbff' }
                  else if (isCandidate){ borderColor = '#67c23a'; backgroundColor = '#f0f9ff' }
                  return (
                    <div key={`next-${i}`} className='candidate' style={{borderColor, background: backgroundColor, marginBottom:4, cursor:'pointer'}}
                         onClick={()=>{
                           if (isCandidate){
                             const candIdx = candList.findIndex(c => c.seg_id===origSeg.seg_id && c.scene_id===origSeg.scene_id)
                             if (candIdx>=0) seekToCandidate(candList[candIdx], candIdx); else seekToOrigSegment(origSeg)
                           }else{ seekToOrigSegment(origSeg) }
                         }}>
                      <div style={{display:'flex',justifyContent:'space-between',marginBottom:4}}>
                        <div style={{ fontWeight: isPlaying ? 'bold' : 'normal', color: isPlaying ? '#ff6b35' : '#333' }}>
                          {isPlaying && <span style={{ marginRight: 4 }}>▶</span>}
                          seg {origSeg.seg_id} S{origSeg._sceneId} / idx {origSeg.scene_seg_idx}
                          {isCandidate && <span style={{color:'#67c23a', marginLeft:8}}>✓ 候选</span>}
                        </div>
                        <div style={{fontWeight:600}}>{isCandidate ? (candList.find(c => c.seg_id===origSeg.seg_id)?.score?.toFixed?.(3) ?? '-') : '-'}</div>
                      </div>
                      <div style={{fontSize:12, opacity: isPlaying?1:.7, fontWeight: isPlaying?'bold':'normal', color: isPlaying?'#ff6b35':'inherit'}}>
                        {formatTime(origSeg.start ?? 0)} - {formatTime(origSeg.end ?? 0)}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          ) : (
            // === 其他：候选列表 ===
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
                    seg{c.seg_id} S{c.scene_id} / idx {c.scene_seg_idx}
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
                const firstOrigSeg = origSegments[0]
                if (firstOrigSeg) { setSelectedCandIdx(0); seekToOrigSegment(firstOrigSeg) }
              } else if (candMode === 'corridor') {
                const first = corridorPrev[0] || corridorNext[0]
                if (first) { setSelectedCandIdx(0); seekToOrigSegment(first) }
              } else if (candList && candList.length > 0) {
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