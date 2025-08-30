import React, {useEffect, useMemo, useRef, useState} from 'react'
import {openProject,listScenes,listSegments,applyChanges,save,rebuildScene, listOrigSegments} from '../api'
import { formatTime, usePersistedState, useCandidates, usePlayerController, type Scene, type SegmentRow } from './logic'

/**
 * Sidecar-only 播放模式 + Debug 强化：
 *  - 完全不使用 <video>，只走 sidecar（mpv）。
 *  - 未检测到 sidecar 直接报错并在 UI 顶部显著提示。
 *  - 加强日志：ErrorBoundary、window.onerror、unhandledrejection、重要流程 console.log。
 */

const BACKEND_BASE = `${window.location.protocol}//${window.location.hostname}:8787`
const SIDECAR_BASE = `${window.location.protocol}//${window.location.hostname}:9777`

// 简易日志工具
const L = (...args:any[]) => console.log('%c[App]', 'color:#673ab7', ...args)
const LW = (...args:any[]) => console.warn('%c[App]', 'color:#f57c00', ...args)
const LE = (...args:any[]) => console.error('%c[App]', 'color:#d32f2f', ...args)

// ============ Error Boundary ============

class ErrorBoundary extends React.Component<{children: React.ReactNode}, {error:any, info:any}> {
  constructor(props:any){ super(props); this.state = { error:null, info:null } }
  static getDerivedStateFromError(error:any){ return { error, info: null } }
  componentDidCatch(error:any, info:any){ LE('[ErrorBoundary] Render error', error, info); this.setState({ error, info }) }
  render(){
    if (this.state.error){
      return (
        <div style={{background:'#fdecea', color:'#c62828', padding:'12px 16px', border:'1px solid #f5c6cb', borderRadius:6, margin:12}}>
          <div style={{fontWeight:700}}>页面渲染异常（ErrorBoundary 捕获）</div>
          <div style={{fontFamily:'monospace', fontSize:12, whiteSpace:'pre-wrap', marginTop:6}}>
            {String(this.state.error)}
          </div>
          <div style={{fontSize:12, opacity:.8, marginTop:6}}>详细堆栈请查看浏览器控制台（Console）。</div>
        </div>
      )
    }
    return this.props.children as any
  }
}

// ============ Sidecar helper ============

async function sidecarOpen(moviePath: string, clipPath: string) {
  if (!moviePath || !clipPath) {
    LW('[sidecar] movie/clip path empty, skip open', { moviePath, clipPath })
    return
  }
  try {
    L('[sidecar] POST /open', { moviePath, clipPath })
    const resp = await fetch(`${SIDECAR_BASE}/open`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ movie: moviePath, clip: clipPath }),
    })
    const j = await resp.json()
    L('[sidecar] open ->', j)
  } catch (e) {
    LE('[sidecar] open failed', e)
  }
}

export default function App(){
  L('App() render begin')

  // ============ Global runtime error hooks ============
  const [lastRuntimeError, setLastRuntimeError] = useState<string | null>(null)
  useEffect(()=>{
    const onErr = (e:ErrorEvent)=>{ LE('[window.error]', e.message, e.error); setLastRuntimeError(`error: ${e.message}`) }
    const onRej = (e:PromiseRejectionEvent)=>{ LE('[unhandledrejection]', e.reason); setLastRuntimeError(`unhandledrejection: ${String(e.reason)}`) }
    window.addEventListener('error', onErr)
    window.addEventListener('unhandledrejection', onRej)
    return ()=>{ window.removeEventListener('error', onErr); window.removeEventListener('unhandledrejection', onRej) }
  },[])

  const [root, setRoot]   = usePersistedState<string>('rm_root', '')
  const [movie, setMovie] = usePersistedState<string>('rm_movie', '')
  const [clip, setClip]   = usePersistedState<string>('rm_clip', '')

  // --- Sidecar 状态 ---
  const [sidecarOk, setSidecarOk] = useState<boolean>(false)
  const [sidecarChecked, setSidecarChecked] = useState<boolean>(false)
  const [sidecarStatusSnap, setSidecarStatusSnap] = useState<any>(null)

  async function checkSidecar(tag:string){
    try {
      L(`[sidecar] GET /status (${tag})`)
      const r = await fetch(`${SIDECAR_BASE}/status`, { cache: 'no-store' })
      const j = await r.json()
      setSidecarOk(Boolean(j && j.ok))
      setSidecarChecked(true)
      setSidecarStatusSnap(j)
      L('[sidecar] status -> ok=', j?.ok, 'playing=', j?.playing, 'ranges=', j?.ranges)
    } catch (e) {
      LE('[sidecar] status failed', e)
      setSidecarOk(false)
      setSidecarChecked(true)
    }
  }

  useEffect(()=>{ checkSidecar('mount'); }, [])

  const [scenes,setScenes]=useState<Scene[]>([])
  const [activeScene,setActiveScene]=useState<number|null>(null)
  const [segments,setSegments]=useState<SegmentRow[]>([])
  const [allSegments,setAllSegments]=useState<SegmentRow[]>([])
  const [selectedSegId,setSelectedSegId]=useState<number|null>(null)
  const [selectedCandIdx,setSelectedCandIdx]=useState<number>(0)

  const [pendingChoice, setPendingChoice] = useState<{type:'cand'|'orig', data:any} | null>(null)
  const [selectedMo, setSelectedMo] = useState<any|null>(null)

  const [followMovie,setFollowMovie]=useState<boolean>(true)
  const [loop,setLoop]=useState<boolean>(true)
  const [mirrorClip, setMirrorClip] = useState<boolean>(false) // 占位
  const [candMode, setCandMode] = useState<'top'|'scene'|'all'|'corridor'>('top')
  const { items: candList, total: candTotal } = useCandidates(selectedSegId, candMode, 120)

  const [origSegments, setOrigSegments] = useState<any[]>([])
  const [showOrigSegments, setShowOrigSegments] = useState<boolean>(false)

  const [corridorPrev, setCorridorPrev] = useState<any[]>([])
  const [corridorNext, setCorridorNext] = useState<any[]>([])
  const corridorCacheRef = useRef(new Map<number, any[]>())
  const CORRIDOR_N = 2

  // 仅为 usePlayerController 形参占位，不使用 <video>
  const clipRef = useRef<HTMLVideoElement|null>(null)
  const movieRef = useRef<HTMLVideoElement|null>(null)

  const [debug, setDebug] = useState<boolean>(true)
  const [showDebugPanel, setShowDebugPanel] = useState<boolean>(true)
  const [overrides, setOverrides] = useState<any>(null)
  const [sceneHints, setSceneHints] = useState<Record<number, {scene_id:number, scene_seg_idx:number} | null>>({})

  const getOverrideForSeg = (segId: number) => {
    const d: any = overrides?.data || {}
    return d[segId] ?? d[String(segId)] ?? null
  }

  async function refreshOverrides(){
    try{
      L('[overrides] GET /overrides')
      const resp = await fetch(`${BACKEND_BASE}/overrides`)
      const j = await resp.json()
      setOverrides(j)
      L('[overrides] count=', j?.count, 'path=', j?.path)
    }catch(e){ LE('load overrides failed', e) }
  }
  async function clearOverrides(){
    try{
      L('[overrides] POST /overrides/clear')
      await fetch(`${BACKEND_BASE}/overrides/clear`, {method:'POST'})
      await refreshOverrides()
    }catch(e){ LE('clear overrides failed', e) }
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
      L('[hint] scene', sceneId, '->', hint)
      setSceneHints(prev=> ({ ...prev, [sceneId]: hint }))
    }catch(e){ LE('ensureSceneHint failed', e) }
  }

  useEffect(()=>{
    if (!Array.isArray(scenes) || scenes.length===0) return
    const firstIds = scenes.slice(0, Math.min(20, scenes.length)).map(s=>s.clip_scene_id)
    firstIds.forEach(id=>{ ensureSceneHint(id) })
  }, [scenes])

  // 播放器同步（sidecar 内部实际完成）
  const [syncPlay, setSyncPlay] = useState<boolean>(true)
  const [maxLoops, setMaxLoops] = useState<number>(1)
  const { isPlaying, loopCount, range, playPair, play: playSync, pause: pauseSync } =
    usePlayerController({
      clipRef,
      movieRef,
      backendBase: BACKEND_BASE,
      syncPlay,
      maxLoops: (loop ? maxLoops : 1),
      debug: true, // 强制打印 hook 内部关键日志
      onSetSrc: (_cUrl, _mUrl) => { /* no-op */ }
    })

  // 播放态 / 当前行
  const [playingSegId, setPlayingSegId] = useState<number | null>(null)
  const [playingFromSide, setPlayingFromSide] = useState<'left' | 'right' | null>(null)
  const [playingType, setPlayingType] = useState<'clip' | 'orig' | null>(null)
  const [playingOrigSegId, setPlayingOrigSegId] = useState<number | null>(null)

  // 当前校对的 clip 段
  const [currentClipSegment, setCurrentClipSegment] = useState<{
    segId: number,
    clipStart: number,
    clipEnd: number,
    sceneId?: number,
    sceneSegIdx?: number
  } | null>(null)

  // 最近一次 seek 来源
  const lastSeekByRef = useRef<'left'|'right'|'auto'|null>(null)

  async function refreshScenes(){
    try {
      L('[scenes] listScenes() begin')
      const arr: Scene[] = await listScenes()
      L('[scenes] listScenes() ->', Array.isArray(arr)?arr.length:arr)
      setScenes(arr)
      const allSegs: SegmentRow[] = []
      for (const scene of arr) {
        try {
          const segs = await listSegments(scene.clip_scene_id)
          L(`[segments] scene ${scene.clip_scene_id} -> ${segs.length}`)
          allSegs.push(...segs)
        } catch (e) {
          LE(`Failed to load segments for scene ${scene.clip_scene_id}`, e)
        }
      }
      L('[segments] total ->', allSegs.length)
      setAllSegments(allSegs)
      if (allSegs && allSegs.length) {
        const firstSeg = allSegs[0]
        setSelectedSegId(firstSeg.seg_id)
        setCurrentClipSegment({
          segId: firstSeg.seg_id,
          clipStart: firstSeg.clip.start ?? 0,
          clipEnd: firstSeg.clip.end ?? 0,
          sceneId: firstSeg.clip.scene_id,
          sceneSegIdx: firstSeg.clip.scene_seg_idx
        })
        L('[select] init seg ->', firstSeg.seg_id)
      }
    } catch (e) {
      LE('listScenes failed', e)
    }
  }

  async function doOpen(){
    try {
      L('[open] openProject()', { root, movie, clip })
      await openProject(root, movie||undefined, clip||undefined)
      L('[open] openProject() done')
    } catch (e) {
      LE('openProject failed', e)
    }
    await sidecarOpen(movie, clip)
    await checkSidecar('after-open')
    if (!sidecarOk) {
      LW('Sidecar 未就绪（禁止回退）')
      alert('Sidecar (mpv) 未就绪或未响应，已禁用浏览器 <video> 回退。\n请先在本机启动: python src/sidecar/app.py')
    }
    await Promise.all([
      refreshOverrides(),
      refreshScenes(),
    ])
  }

  useEffect(() => {
    if (movie && clip) {
      L('[open] auto sidecarOpen() due to movie/clip change')
      sidecarOpen(movie, clip)
      checkSidecar('movie/clip-change')
    }
  }, [movie, clip])

  // selection -> load scene rows
  useEffect(()=>{
    if(selectedSegId != null){
      const selectedSeg = allSegments.find(s => s.seg_id === selectedSegId)
      if (selectedSeg) {
        setActiveScene(selectedSeg.clip.scene_id || null)
        const merged = (selectedSeg as any).matched_orig_seg || null
        setSelectedMo(merged)
        if (selectedSeg.clip.scene_id != null) {
          listSegments(selectedSeg.clip.scene_id).then(arr=>{
            L('[segments] load for activeScene', selectedSeg.clip.scene_id, '->', arr.length)
            setSegments(arr)
          }).catch(e => {
            LE('Failed to load segments', e)
          })
        }
      }
    }
  },[selectedSegId, allSegments])

  // 加载原始段落数据 (当前场景及前后场景)
  async function loadOrigSegments(sceneId: number) {
    try {
      L('[orig] load around scene', sceneId)
      const promises:any[] = []
      const sceneIds:number[] = []
      for (let i = sceneId - 2; i <= sceneId + 2; i++) {
        if (i > 0) { promises.push(listOrigSegments(i)); sceneIds.push(i) }
      }
      const responses = await Promise.all(promises)
      const merged:any[] = []
      responses.forEach((resp, idx) => {
        const segs = resp?.segments || []
        L('[orig] scene', sceneIds[idx], '->', segs.length)
        segs.forEach((seg:any)=> merged.push({ ...seg, _sceneId: sceneIds[idx], _isCurrentScene: sceneIds[idx]===sceneId }))
      })
      merged.sort((a,b)=> a._sceneId!==b._sceneId ? a._sceneId-b._sceneId : (a.scene_seg_idx||0)-(b.scene_seg_idx||0))
      setOrigSegments(merged)
      L('[orig] merged ->', merged.length)
    } catch (e) {
      LE('Failed to load orig segments', e)
      setOrigSegments([])
    }
  }

  // 加载走廊
  async function loadCorridorFor(anchorSceneId: number, dir: 'prev'|'next', n = CORRIDOR_N){
    const ids: number[] = []
    if (dir === 'prev'){ for (let s = anchorSceneId + 1; s <= anchorSceneId + n; s++) ids.push(s) }
    else { for (let s = anchorSceneId - n; s <= anchorSceneId - 1; s++) if (s > 0) ids.push(s) }
    const all: any[] = []
    for (const sid of ids){
      if (!corridorCacheRef.current.has(sid)){
        try{
          const resp = await listOrigSegments(sid)
          const arr = (resp?.segments || []).map((x:any)=>({ ...x, _sceneId: sid, _corridor: dir }))
          corridorCacheRef.current.set(sid, arr)
        }catch(e){ LE('loadCorridorFor failed', e); corridorCacheRef.current.set(sid, []) }
      }
      all.push(...(corridorCacheRef.current.get(sid) || []))
    }
    L(`[corridor] dir=${dir} anchor=${anchorSceneId} ->`, all.length)
    if (dir==='prev') setCorridorPrev(all); else setCorridorNext(all)
  }

  // derive selected row
  const selectedRow: SegmentRow | undefined = useMemo(()=>{
    const row = allSegments.find(s=>s.seg_id===selectedSegId) || segments.find(s=>s.seg_id===selectedSegId)
    return row
  },[allSegments, segments, selectedSegId])

  useEffect(() => {
    if (candMode === 'scene' && selectedRow?.matched_orig_seg?.scene_id) {
      loadOrigSegments(selectedRow.matched_orig_seg.scene_id)
      setShowOrigSegments(true)
    } else {
      setShowOrigSegments(false)
    }
  }, [candMode, selectedRow])

  // 基于相邻 clip 场景的锚点 scene，加载走廊
  useEffect(()=>{
    if (!selectedRow) return
    const cid = selectedRow.clip?.scene_id
    if (cid == null) return
    const sceneIds = Array.from(new Set(allSegments.map(s=>s.clip?.scene_id).filter((x:any)=>x!=null))).sort((a:any,b:any)=>a-b)
    const idx = sceneIds.indexOf(cid)
    const prevCid = idx>0 ? sceneIds[idx-1] : null
    const nextCid = (idx>=0 && idx<sceneIds.length-1) ? sceneIds[idx+1] : null
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

  // 将当前选中行定位到 sidecar（clip=行时间，movie=候选/匹配）
  function seekTo(row?: SegmentRow, candIdx?: number){
    if (!sidecarOk) { LE('[seekTo] sidecar not ready, skip'); return }
    const r = row || selectedRow
    if (!r) return
    const clipStart = r.clip.start ?? 0
    const clipEnd   = r.clip.end   ?? (clipStart + 2)
    let mo: any = selectedMo || (r as any).matched_orig_seg || {}
    const cand = (candList && candList[candIdx ?? selectedCandIdx]) || ((r.top_matches && r.top_matches[candIdx ?? selectedCandIdx]) || null)
    const hasOv = !!(r as any).is_override
    if (!selectedMo && followMovie && !hasOv && cand) mo = cand
    const movStart = mo?.start ?? 0
    const movEnd   = mo?.end   ?? (movStart + 2)
    L('[seekTo] playPair', { clipStart, clipEnd, movStart, movEnd, segId:r.seg_id })
    playPair(clipStart, clipEnd, movStart, movEnd)
    setPlayingSegId(r.seg_id)
    setPlayingFromSide('left')
    setPlayingType('clip')
    setPlayingOrigSegId(mo?.seg_id ?? null)
    setCurrentClipSegment({ segId: r.seg_id, clipStart, clipEnd, sceneId: r.clip.scene_id, sceneSegIdx: r.clip.scene_seg_idx })
  }

  function seekFromLeft(seg: SegmentRow){
    if (!sidecarOk) { alert('Sidecar 未就绪，无法播放'); return }
    const clipStart = seg.clip?.start ?? 0
    const clipEnd   = seg.clip?.end   ?? (clipStart + 2)
    const mo: any = (seg as any).matched_orig_seg || {}
    setSelectedMo(mo)
    const movStart = mo?.start ?? 0
    const movEnd   = mo?.end   ?? (movStart + 2)
    L('[seekFromLeft] playPair', { clipStart, clipEnd, movStart, movEnd, segId: seg.seg_id })
    lastSeekByRef.current = 'left'
    playPair(clipStart, clipEnd, movStart, movEnd)
    setSelectedSegId(seg.seg_id)
    setPlayingSegId(seg.seg_id)
    setPlayingFromSide('left')
    setPlayingType('clip')
    setPlayingOrigSegId(mo?.seg_id ?? null)
    setPendingChoice(null)
    setCurrentClipSegment({ segId: seg.seg_id, clipStart, clipEnd, sceneId: seg.clip.scene_id, sceneSegIdx: seg.clip.scene_seg_idx })
  }

  function seekToOrigSegment(origSeg: any) {
    if (!sidecarOk) { alert('Sidecar 未就绪，无法播放'); return }
    if (!origSeg || !currentClipSegment) return
    const clipStart = currentClipSegment.clipStart
    const clipEnd   = currentClipSegment.clipEnd
    const movStart  = origSeg.start ?? 0
    const movEnd    = origSeg.end   ?? (movStart + 2)
    L('[seekToOrigSegment] playPair', { clipStart, clipEnd, movStart, movEnd, segId: currentClipSegment.segId, origSegId: origSeg.seg_id })
    lastSeekByRef.current = 'right'
    playPair(clipStart, clipEnd, movStart, movEnd)
    setPlayingSegId(currentClipSegment.segId)
    setPlayingFromSide('right')
    setPlayingType('orig')
    setPlayingOrigSegId(origSeg.seg_id ?? null)
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
    if (!sidecarOk) { alert('Sidecar 未就绪，无法播放'); return }
    if (!currentClipSegment || !candidate) return
    const clipStart = currentClipSegment.clipStart
    const clipEnd   = currentClipSegment.clipEnd
    const movStart  = candidate.start ?? 0
    const movEnd    = candidate.end   ?? (movStart + 2)
    L('[seekToCandidate] playPair', { clipStart, clipEnd, movStart, movEnd, segId: currentClipSegment.segId, candIdx })
    setSelectedCandIdx(candIdx)
    setPendingChoice({ type: 'cand', data: candidate })
    setSelectedMo(candidate || null)
    playPair(clipStart, clipEnd, movStart, movEnd)
    setPlayingSegId(currentClipSegment.segId)
    setPlayingFromSide('right')
    setPlayingType('clip')
    setPlayingOrigSegId(candidate.seg_id ?? null)
  }

  const currentCandKey = useMemo(()=>{
    const c:any = candList?.[selectedCandIdx]
    return c ? `${c.seg_id}-${c.start}-${c.end}` : 'none'
  }, [candList, selectedCandIdx])

  useEffect(()=>{
    if (!sidecarOk) { LW('[autoseek] skip because sidecar not ok'); return }
    if (lastSeekByRef.current){ L('[autoseek] skip once due to lastSeekByRef', lastSeekByRef.current); lastSeekByRef.current = null; return }
    L('[autoseek] trigger', { selectedSegId, selectedCandIdx, followMovie, currentCandKey })
    seekTo()
  },[selectedSegId, selectedCandIdx, followMovie, currentCandKey, sidecarOk])

  // 应用候选
  async function acceptSelected(){
    const r = selectedRow
    if (!r) { LW('[apply] no selected row'); return }
    const fromPending = pendingChoice?.data
    const fromCandList = (candList && candList[selectedCandIdx]) || null
    const fromRowTop = (r.top_matches && r.top_matches[selectedCandIdx]) || null
    const chosen = fromPending || fromCandList || fromRowTop
    if (!chosen){ LW('[apply] no candidate to apply'); return }
    const change = { seg_id: r.seg_id, chosen }
    try{
      L('[apply] sending change', change)
      await applyChanges([change])
      setSegments(prev => Array.isArray(prev) ? prev.map(row => row.seg_id===r.seg_id ? ({...row, matched_orig_seg: {...chosen}, is_override: true, matched_source: 'applied'}) : row) : prev)
      setAllSegments(prev => Array.isArray(prev) ? prev.map(row => row.seg_id===r.seg_id ? ({...row, matched_orig_seg: {...chosen}, is_override: true, matched_source: 'applied'}) : row) : prev)
      setOverrides(prev => {
        const data = { ...(prev?.data || {}), [String(r.seg_id)]: { ...chosen } }
        return { ...(prev || {}), data, count: Object.keys(data).length }
      })
      setSelectedMo(chosen)
      L('[apply] applied on seg', r.seg_id, '->', chosen)
      await refreshOverrides()
    } catch (e:any) {
      LE('[apply] failed', e)
      alert('应用失败: ' + (e?.message || String(e)))
    }
  }

  // 场景级重建
  async function doRebuild(){
    if (activeScene==null) return
    L('[rebuild] scene', activeScene)
    await rebuildScene(activeScene)
    await refreshScenes()
    await refreshOverrides()
  }

  // ============ UI ============

  return (
    <ErrorBoundary>
      {/* 顶部 Debug 条 */}
      <div style={{
        position:'sticky', top:0, zIndex:999,
        background:'#111', color:'#eee', padding:'6px 10px', fontSize:12, display:'flex', gap:12, alignItems:'center'
      }}>
        <span>UI Boot OK</span>
        <span>sidecar: <b style={{color: sidecarOk? '#8bc34a':'#ff5252'}}>{sidecarOk? 'OK':'OFF'}</b></span>
        <span>seg: <b>{selectedSegId ?? '-'}</b></span>
        <span>playing: <b>{String(isPlaying)}</b></span>
        {range && (
          <span>range: C[{formatTime(range.clipStart)}~{formatTime(range.clipEnd)}] / M[{formatTime(range.movieStart)}~{formatTime(range.movieEnd)}]</span>
        )}
        {lastRuntimeError && <span style={{color:'#ff8a80'}}>ERR: {lastRuntimeError}</span>}
        <span style={{marginLeft:'auto', opacity:.7}}>root={root || '(unset)'} / movie={movie ? '✓':'-'} / clip={clip ? '✓':'-'}</span>
      </div>

      <div className='layout'>
        <div className='toolbar'>
          <input style={{width:320}} placeholder='project root' value={root} onChange={e=>{ setRoot(e.target.value) }} />
          <input style={{width:260}} placeholder='movie.mp4 (absolute path)' value={movie} onChange={e=>{ setMovie(e.target.value) }} />
          <input style={{width:260}} placeholder='clip.mp4 (absolute path)' value={clip} onChange={e=>{ setClip(e.target.value) }} />
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
          <span style={{marginLeft:12, padding:'2px 6px', borderRadius:4, fontSize:12, background: sidecarOk? '#e8f5e9':'#fdecea', color: sidecarOk? '#2e7d32':'#c62828'}}>
            sidecar: {sidecarOk? 'OK':'OFF'}
          </span>
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
                      onClick={()=>{ L('[UI] click left seg', seg.seg_id); setSelectedSegId(seg.seg_id); seekFromLeft(seg) }}
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
                <button onClick={()=>{ if (!sidecarOk) { alert('Sidecar 未就绪'); return } L('[UI] click play/pause'); (isPlaying ? pauseSync() : playSync()) }}>
                  {isPlaying ? '⏸️ 暂停' : '▶️ 播放'}
                </button>
                <label style={{display: 'flex', alignItems: 'center', gap: 4}}>
                  <input 
                    type="checkbox" 
                    checked={syncPlay} 
                    onChange={(e) => { L('[UI] toggle syncPlay', e.target.checked); setSyncPlay(e.target.checked) }} 
                  />
                  同步播放（sidecar）
                </label>
              </div>
            </div>
            
            {/* Sidecar-only 播放信息占位（不渲染 <video>） */}
            <div className='videos'>
              <div style={{width:'100%', padding:'8px 12px', border:'1px dashed #ddd', borderRadius:6, background:'#fafafa'}}>
                {(!sidecarChecked || sidecarOk) ? (
                  <div style={{fontSize:13}}>
                    <div style={{marginBottom:6}}>🎬 正在使用 <b>mpv sidecar</b> 播放（浏览器不渲染 <code>&lt;video&gt;</code>）。</div>
                    {range ? (
                      <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:12}}>
                        <div>
                          <div style={{fontSize:12, opacity:.7}}>Clip 区间</div>
                          <div style={{fontWeight:600}}>{formatTime(range.clipStart)} – {formatTime(range.clipEnd)}</div>
                        </div>
                        <div>
                          <div style={{fontSize:12, opacity:.7}}>Movie 区间</div>
                          <div style={{fontWeight:600}}>{formatTime(range.movieStart)} – {formatTime(range.movieEnd)}</div>
                        </div>
                      </div>
                    ) : (
                      <div style={{fontSize:12, opacity:.7}}>尚未选择段落</div>
                    )}
                    {sidecarStatusSnap && (
                      <div style={{marginTop:8, fontSize:12, opacity:.8}}>
                        <div>sidecar.playing = {String(sidecarStatusSnap.playing)}</div>
                        <div>sidecar.ranges = {JSON.stringify(sidecarStatusSnap.ranges)}</div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div style={{color:'#c00', fontWeight:600}}>
                    ⚠️ Sidecar 未运行或不可达（无浏览器回退）。
                    <div style={{fontSize:12, opacity:.8, marginTop:6}}>请执行：<code>python src/sidecar/app.py</code>，然后点击“打开”。</div>
                  </div>
                )}
              </div>
            </div>

            <table className='seg-table'>
              <thead><tr><th>seg_id</th><th>clip idx</th><th>clip t</th><th>orig t</th><th>matched scene/idx</th><th>score</th><th>操作</th></tr></thead>
              <tbody>
                {segments.map((s:SegmentRow)=>{
                  const mo: any = (s as any).matched_orig_seg || {}
                  const clipStart = s.clip.start ?? 0
                  const clipEnd = s.clip.end ?? 0
                  const clipTime = `${formatTime(clipStart)} - ${formatTime(clipEnd)}`
                  const origStart = mo.start ?? 0
                  const origEnd = mo.end ?? 0
                  const origTime = mo.start !== undefined ? `${formatTime(origStart)} - ${formatTime(origEnd)}` : '-'
                  const isSel = selectedSegId===s.seg_id
                  const isPlayingRow = playingSegId === s.seg_id && playingFromSide === 'left'
                  let bgColor = 'transparent'
                  if (isPlayingRow) { bgColor = '#fff5f2' }
                  else if (isSel) { bgColor = '#f7fbff' }
                  return <tr key={s.seg_id}
                            style={{
                              background: bgColor, 
                              cursor:'pointer',
                              borderLeft: isPlayingRow ? '4px solid #ff6b35' : isSel ? '4px solid #409eff' : '4px solid transparent'
                            }}
                            onClick={()=>{ 
                              L('[UI] click row seg', s.seg_id)
                              setSelectedSegId(s.seg_id); 
                              setSelectedCandIdx(0); 
                              setPendingChoice(null);
                              setCurrentClipSegment({
                                segId: s.seg_id,
                                clipStart: s.clip.start ?? 0,
                                clipEnd: s.clip.end ?? 0,
                                sceneId: s.clip.scene_id,
                                sceneSegIdx: s.clip.scene_seg_idx
                              });
                              seekTo(s,0);
                            } }>
                    <td style={{ fontWeight: isPlayingRow ? 'bold' : 'normal', color: isPlayingRow ? '#ff6b35' : 'inherit' }}>
                      {isPlayingRow && <span style={{ marginRight: 4 }}>▶</span>}
                      {s.seg_id}
                    </td>
                    <td>{s.clip.scene_seg_idx}</td>
                    <td style={{ fontSize: '12px' }}>{clipTime}</td>
                    <td style={{ fontSize: '12px' }}>{origTime}</td>
                    <td>seg{mo.seg_id??'-'} S{mo.scene_id??'-'} / idx {mo.scene_seg_idx??'-'}</td>
                    <td>{(mo.score??0).toFixed(3)}</td>
                    <td>
                      <button onClick={(e)=>{ e.stopPropagation(); L('[UI] click apply seg', s.seg_id); acceptSelected() }}>接受候选</button>
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
                          onClick={()=>{ L('[UI] candMode ->', md); setCandMode(md) }}
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
                      const sceneGroups = new Map()
                      origSegments.forEach(seg => {
                        const sceneId = seg._sceneId
                        if (!sceneGroups.has(sceneId)) sceneGroups.set(sceneId, [])
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
                            const isCandidate = candList.some(c => c.seg_id === origSeg.seg_id && c.scene_id === origSeg.scene_id)
                            const isSelected = candList[selectedCandIdx]?.seg_id === origSeg.seg_id
                            const isPlayingOrig = playingType === 'orig' && playingOrigSegId === origSeg.seg_id
                            let borderColor = '#eee'
                            let backgroundColor = '#fff'
                            if (isPlayingOrig) { borderColor = '#ff6b35'; backgroundColor = '#fff5f2' }
                            else if (isSelected) { borderColor = '#409eff'; backgroundColor = '#f5fbff' }
                            else if (isCandidate) { borderColor = '#67c23a'; backgroundColor = '#f0f9ff' }
                            
                            return (
                              <div key={`${sceneId}-${i}`} 
                                  className='candidate'
                                  style={{
                                    borderColor,
                                    background: backgroundColor,
                                    opacity: origSeg._isCurrentScene ? (isCandidate ? 1 : 0.8) : (isCandidate ? 1 : 0.4),
                                    cursor: 'pointer',
                                    marginBottom: 4,
                                    boxShadow: isPlayingOrig ? '0 0 8px rgba(255, 107, 53, 0.3)' : undefined
                                  }}
                                  onClick={() => {
                                    L('[UI] click origSeg', { sceneId, segId: origSeg.seg_id })
                                    if (isCandidate) {
                                      const candIdx = candList.findIndex(c => c.seg_id === origSeg.seg_id && c.scene_id === origSeg.scene_id)
                                      if (candIdx >= 0) seekToCandidate(candList[candIdx], candIdx)
                                      else seekToOrigSegment(origSeg)
                                    } else {
                                      seekToOrigSegment(origSeg)
                                    }
                                  }}>
                                <div style={{display:'flex',justifyContent:'space-between',marginBottom:4}}>
                                  <div style={{ fontWeight: isPlayingOrig ? 'bold' : 'normal', color: isPlayingOrig ? '#ff6b35' : '#333' }}>
                                    {isPlayingOrig && <span style={{ marginRight: 4, color: '#ff6b35' }}>▶</span>}
                                    seg {origSeg.seg_id} S{origSeg._sceneId} / idx {origSeg.scene_seg_idx}
                                    {isCandidate && <span style={{color:'#67c23a', marginLeft:8}}>✓ 候选</span>}
                                    {!origSeg._isCurrentScene && <span style={{color:'#999', marginLeft:8, fontSize:11}}>其他场景</span>}
                                  </div>
                                  <div style={{fontWeight:600}}>
                                    {isCandidate ? (candList.find(c => c.seg_id === origSeg.seg_id)?.score?.toFixed?.(3) ?? '-') : '-'}
                                  </div>
                                </div>
                                <div style={{fontSize:12, opacity: isPlayingOrig ? 1 : .7, fontWeight: isPlayingOrig ? 'bold' : 'normal', color: isPlayingOrig ? '#ff6b35' : 'inherit'}}>
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
                      const isPlayingOrig = playingType === 'orig' && playingOrigSegId === origSeg.seg_id
                      let borderColor = '#eee', backgroundColor = '#fff'
                      if (isPlayingOrig){ borderColor = '#ff6b35'; backgroundColor = '#fff5f2' }
                      else if (isSelected){ borderColor = '#409eff'; backgroundColor = '#f5fbff' }
                      else if (isCandidate){ borderColor = '#67c23a'; backgroundColor = '#f0f9ff' }
                      return (
                        <div key={`prev-${i}`} className='candidate' style={{borderColor, background: backgroundColor, marginBottom:4, cursor:'pointer'}}
                              onClick={()=>{
                                L('[UI] click corridor prev orig', origSeg.seg_id)
                                if (isCandidate){
                                  const candIdx = candList.findIndex(c => c.seg_id===origSeg.seg_id && c.scene_id===origSeg.scene_id)
                                  if (candIdx>=0) seekToCandidate(candList[candIdx], candIdx); else seekToOrigSegment(origSeg)
                                }else{ seekToOrigSegment(origSeg) }
                              }}>
                          <div style={{display:'flex',justifyContent:'space-between',marginBottom:4}}>
                            <div style={{ fontWeight: isPlayingOrig ? 'bold' : 'normal', color: isPlayingOrig ? '#ff6b35' : '#333' }}>
                              {isPlayingOrig && <span style={{ marginRight: 4 }}>▶</span>}
                              seg {origSeg.seg_id} S{origSeg._sceneId} / idx {origSeg.scene_seg_idx}
                              {isCandidate && <span style={{color:'#67c23a', marginLeft:8}}>✓ 候选</span>}
                            </div>
                            <div style={{fontWeight:600}}>{isCandidate ? (candList.find(c => c.seg_id===origSeg.seg_id)?.score?.toFixed?.(3) ?? '-') : '-'}</div>
                          </div>
                          <div style={{fontSize:12, opacity: isPlayingOrig?1:.7, fontWeight: isPlayingOrig?'bold':'normal', color: isPlayingOrig?'#ff6b35':'inherit'}}>
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
                      const isPlayingOrig = playingType === 'orig' && playingOrigSegId === origSeg.seg_id
                      let borderColor = '#eee', backgroundColor = '#fff'
                      if (isPlayingOrig){ borderColor = '#ff6b35'; backgroundColor = '#fff5f2' }
                      else if (isSelected){ borderColor = '#409eff'; backgroundColor = '#f5fbff' }
                      else if (isCandidate){ borderColor = '#67c23a'; backgroundColor = '#f0f9ff' }
                      return (
                        <div key={`next-${i}`} className='candidate' style={{borderColor, background: backgroundColor, marginBottom:4, cursor:'pointer'}}
                              onClick={()=>{
                                L('[UI] click corridor next orig', origSeg.seg_id)
                                if (isCandidate){
                                  const candIdx = candList.findIndex(c => c.seg_id===origSeg.seg_id && c.scene_id===origSeg.scene_id)
                                  if (candIdx>=0) seekToCandidate(candList[candIdx], candIdx); else seekToOrigSegment(origSeg)
                                }else{ seekToOrigSegment(origSeg) }
                              }}>
                          <div style={{display:'flex',justifyContent:'space-between',marginBottom:4}}>
                            <div style={{ fontWeight: isPlayingOrig ? 'bold' : 'normal', color: isPlayingOrig ? '#ff6b35' : '#333' }}>
                              {isPlayingOrig && <span style={{ marginRight: 4 }}>▶</span>}
                              seg {origSeg.seg_id} S{origSeg._sceneId} / idx {origSeg.scene_seg_idx}
                              {isCandidate && <span style={{color:'#67c23a', marginLeft:8}}>✓ 候选</span>}
                            </div>
                            <div style={{fontWeight:600}}>{isCandidate ? (candList.find(c => c.seg_id===origSeg.seg_id)?.score?.toFixed?.(3) ?? '-') : '-'}</div>
                          </div>
                          <div style={{fontSize:12, opacity: isPlayingOrig?1:.7, fontWeight: isPlayingOrig?'bold':'normal', color: isPlayingOrig?'#ff6b35':'inherit'}}>
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
                  const isPlayingCand = playingType === 'clip' && playingFromSide === 'right' && isSelected
                  let borderColor = '#eee'
                  let backgroundColor = '#fff'
                  if (isPlayingCand) { borderColor = '#ff6b35'; backgroundColor = '#fff5f2' }
                  else if (isSelected) { borderColor = '#409eff'; backgroundColor = '#f5fbff' }
                  
                  return <div key={i} className='candidate'
                              style={{
                                borderColor, 
                                background: backgroundColor,
                                boxShadow: isPlayingCand ? '0 0 8px rgba(255, 107, 53, 0.3)' : undefined
                              }}
                              onClick={()=>{ L('[UI] click candidate', {i, seg:c.seg_id}); seekToCandidate(c, i) }}>
                    <div style={{display:'flex',justifyContent:'space-between',marginBottom:4}}>
                      <div style={{ fontWeight: isPlayingCand ? 'bold' : 'normal', color: isPlayingCand ? '#ff6b35' : '#333' }}>
                        {isPlayingCand && <span style={{ marginRight: 4 }}>▶</span>}
                        seg{c.seg_id} S{c.scene_id} / idx {c.scene_seg_idx}
                      </div>
                      <div style={{fontWeight:600}}>{(c.score??0).toFixed?.(3) ?? c.score}</div>
                    </div>
                    <div style={{fontSize:12, opacity: isPlayingCand ? 1 : .7, fontWeight: isPlayingCand ? 'bold' : 'normal', color: isPlayingCand ? '#ff6b35' : 'inherit'}}>
                      {formatTime(c.start ?? 0)} - {formatTime(c.end ?? 0)}
                    </div>
                    <div style={{fontSize:12,opacity:.6, marginTop:2}}>src: {c.source || '-'}</div>
                  </div>
                })
              )}
              <div style={{display:'flex', gap:8}}>
                <button onClick={()=>{ L('[UI] click apply'); acceptSelected() }}>应用所选</button>
                <button onClick={()=>{ 
                  L('[UI] click pick-first')
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
                <div style={{marginLeft:12, fontSize:12, opacity:.7}}>sidecar: {sidecarOk ? 'OK' : 'OFF'}</div>
                <div style={{marginLeft:12, fontSize:12, opacity:.7}}>overrides count: {overrides?.count ?? '-'}</div>
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
                <div>
                  <div style={{fontSize:12, opacity:.7, marginBottom:4}}>Sidecar /status 快照</div>
                  <div style={{maxHeight:240, overflow:'auto', border:'1px solid #eee', borderRadius:4, padding:8}}>
                    <pre style={{margin:0, fontSize:12}}>{JSON.stringify(sidecarStatusSnap ?? {}, null, 2)}</pre>
                  </div>
                  <div style={{display:'flex', gap:8, marginTop:8}}>
                    <button onClick={()=>checkSidecar('manual')}>刷新 sidecar /status</button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </ErrorBoundary>
  )
}