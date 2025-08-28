import React, {useEffect, useMemo, useRef, useState} from 'react'
import {openProject,listScenes,listSegments,applyChanges,save,rebuildScene, listCandidates} from '../api'

const BACKEND_BASE = `${window.location.protocol}//${window.location.hostname}:8787`

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
  const [selectedSegId,setSelectedSegId]=useState<number|null>(null)
  const [selectedCandIdx,setSelectedCandIdx]=useState<number>(0)
  const [followMovie,setFollowMovie]=useState<boolean>(true)
  const [loop,setLoop]=useState<boolean>(true)
  const [mirrorClip, setMirrorClip] = useState<boolean>(false)
  const [candMode, setCandMode] = useState<'top'|'scene'|'all'>('top')
  const [candList, setCandList] = useState<any[]>([])

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

  async function refreshScenes(){
    try {
      const arr: Scene[] = await listScenes()
      setScenes(arr)
      if (arr && arr.length) {
        setActiveScene(arr[0].clip_scene_id)
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

  // load segments when scene changes
  useEffect(()=>{
    if(activeScene!=null){
      listSegments(activeScene).then(arr=>{
        setSegments(arr)
        if (arr && arr.length){
          setSelectedSegId(arr[0].seg_id)
          setSelectedCandIdx(0)
          loadCandidates(arr[0].seg_id, candMode)
        } else {
          setSelectedSegId(null)
        }
      })
    }
  },[activeScene])

  // fetch candidates for current seg
  async function loadCandidates(segId: number, mode: 'top'|'scene'|'all'){
    try{
      const resp = await listCandidates(segId, mode, 50, 0)
      const items = resp?.items || []
      setCandList(items)
      setSelectedCandIdx(0)
    }catch(e){
      console.error('listCandidates failed', e)
      const r = segments.find(s=>s.seg_id===segId)
      setCandList(r?.top_matches || [])
      setSelectedCandIdx(0)
    }
  }

  useEffect(()=>{
    if (selectedSegId!=null) loadCandidates(selectedSegId, candMode)
  }, [selectedSegId, candMode])

  // derive selected row
  const selectedRow: SegmentRow | undefined = useMemo(()=>{
    return segments.find(s=>s.seg_id===selectedSegId)
  },[segments,selectedSegId])

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
        try { v.currentTime = s } catch {}
      }
    }
    v.addEventListener('timeupdate', handler)
    ref.current = handler
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

    dlog('seekTo', { seg_id: r.seg_id, clipStart, clipEnd, movStart, movEnd, followMovie })

    const cv = clipRef.current
    const mv = movieRef.current

    // Build URLs with explicit time start; streams start from 0 on the browser timeline
    const clipUrl = `${BACKEND_BASE}/video/clip?t=${clipStart.toFixed(3)}`
    let movieUrl = movieSrc
    let movieIsStream = movieSrc.includes('?t=')
    if (followMovie) {
      movieUrl = `${BACKEND_BASE}/video/movie?t=${movStart.toFixed(3)}`
      movieIsStream = true
    }
    setClipSrc(clipUrl)
    setMovieSrc(movieUrl)

    // When using ?t= streams, the currentTime origin is 0; loop bounds should be relative
    const clipLoopStart = 0
    const clipLoopEnd = Math.max(0.01, (clipEnd - clipStart))
    const movieLoopStart = movieIsStream ? 0 : movStart
    const movieLoopEnd = movieIsStream ? Math.max(0.01, (movEnd - movStart)) : movEnd

    // seek (wait for metadata if needed). For streams, seek to 0; for full files, seek to absolute time
    seekWhenReady(cv, 0)
    seekWhenReady(mv, movieIsStream ? 0 : movStart)

    // optional autoplay (best-effort)
    try { cv && cv.play().catch(()=>{}) } catch {}
    try { mv && mv.play().catch(()=>{}) } catch {}

    // loop handlers with cleanup
    attachLoopSafe(cv, clipLoopStart, clipLoopEnd, clipLoopHandlerRef, clipSeekingRef)
    attachLoopSafe(mv, movieLoopStart, movieLoopEnd, movieLoopHandlerRef, movieSeekingRef)
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
    // reload and advance to next row
    const arr = await listSegments(activeScene!)
    setSegments(arr)
    const idx = arr.findIndex(x=>x.seg_id===r.seg_id)
    const next = idx>=0 && idx+1 < arr.length ? arr[idx+1].seg_id : r.seg_id
    setSelectedSegId(next)
    setSelectedCandIdx(0)
  }

  // Scene level rebuild
  async function doRebuild(){
    if (activeScene==null) return
    await rebuildScene(activeScene)
    const arr = await listSegments(activeScene)
    setSegments(arr)
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
      <label style={{marginRight:12}}><input type='checkbox' checked={mirrorClip} onChange={e=>setMirrorClip(e.target.checked)} /> 镜像Clip</label>
      <label style={{marginRight:12}}><input type='checkbox' checked={debug} onChange={e=>setDebug(e.target.checked)} /> 调试日志</label>
      <label style={{marginRight:12}}><input type='checkbox' checked={showDebugPanel} onChange={e=>setShowDebugPanel(e.target.checked)} /> 显示调试面板</label>
      <button onClick={refreshOverrides}>刷新覆盖</button>
      <button onClick={doRebuild}>场景内重建</button>
      <button onClick={()=>save()}>保存导出</button>
    </div>

    <div className='main'>
      <div className='panel'>
        <div style={{fontWeight:600,marginBottom:6}}>Clip 场景</div>
        <div className='scene-list'>
          {Array.isArray(scenes) && scenes.length===0 && (
            <div style={{fontSize:12,opacity:.7,padding:'8px 4px'}}>无场景数据：检查项目根目录是否包含 match_segments.json / scene_out.json；或点击“刷新场景”。</div>
          )}
          {Array.isArray(scenes) && scenes.map((s:Scene)=>(
            <div key={s.clip_scene_id}
                 className={'scene-item '+(activeScene===s.clip_scene_id?'active':'')}
                 onClick={()=>setActiveScene(s.clip_scene_id)}
                 onMouseEnter={()=>ensureSceneHint(s.clip_scene_id)}>
              <div>#{s.clip_scene_id}</div>
              <div className='badge'>{s.count} / len {s.chain_len}</div>
              <div className='badge' style={{marginLeft:8, opacity: sceneHints[s.clip_scene_id]===undefined? .5 : 1}}>
                {sceneHints[s.clip_scene_id]
                  ? `S${sceneHints[s.clip_scene_id]!.scene_id}:${sceneHints[s.clip_scene_id]!.scene_seg_idx}`
                  : '…'}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className='panel'>
        <div className='videos'>
          <div><div style={{fontSize:12,opacity:.7,marginBottom:4}}>Clip</div><video preload="metadata" ref={clipRef} src={clipSrc} controls
                 style={{ transform: mirrorClip ? 'scaleX(-1)' : undefined, transformOrigin: '50% 50%' }}
                 onLoadedMetadata={e=>{ const v=e.currentTarget as HTMLVideoElement; if (debug) console.log('[Clip] loadedmetadata dur=', v.duration) }}
                 onError={e=>{ console.error('[Clip] error', e) }}
                 onSeeking={e=>{ clipSeekingRef.current = true; if (debug) console.log('[Clip] seeking to', (e.currentTarget as HTMLVideoElement).currentTime) }}
                 onSeeked={e=>{ clipSeekingRef.current = false; if (debug) console.log('[Clip] seeked to', (e.currentTarget as HTMLVideoElement).currentTime) }}
                 onTimeUpdate={e=>{ const now=Date.now(); if (now - clipLastLogRef.current > 1000 && debug){ clipLastLogRef.current = now; console.log('[Clip] t=', (e.currentTarget as HTMLVideoElement).currentTime) } }}
          /></div>
          <div><div style={{fontSize:12,opacity:.7,marginBottom:4}}>Movie</div><video preload="metadata" ref={movieRef} src={movieSrc} controls
                 onLoadedMetadata={e=>{ const v=e.currentTarget as HTMLVideoElement; if (debug) console.log('[Movie] loadedmetadata dur=', v.duration) }}
                 onError={e=>{ console.error('[Movie] error', e) }}
                 onSeeking={e=>{ movieSeekingRef.current = true; if (debug) console.log('[Movie] seeking to', (e.currentTarget as HTMLVideoElement).currentTime) }}
                 onSeeked={e=>{ movieSeekingRef.current = false; if (debug) console.log('[Movie] seeked to', (e.currentTarget as HTMLVideoElement).currentTime) }}
                 onTimeUpdate={e=>{ const now=Date.now(); if (now - movieLastLogRef.current > 1000 && debug){ movieLastLogRef.current = now; console.log('[Movie] t=', (e.currentTarget as HTMLVideoElement).currentTime) } }}
          /></div>
        </div>

        <table className='seg-table'>
          <thead><tr><th>seg_id</th><th>clip idx</th><th>clip t</th><th>matched scene/idx</th><th>score</th><th>操作</th></tr></thead>
          <tbody>
            {segments.map((s:SegmentRow)=>{
              const mo = s.matched_orig_seg || {}
              const t = `${(s.clip.start??0).toFixed(2)}-${(s.clip.end??0).toFixed(2)}`
              const isSel = selectedSegId===s.seg_id
              return <tr key={s.seg_id}
                         style={{background:isSel?'#f7fbff':'transparent', cursor:'pointer'}}
                         onClick={()=>{ setSelectedSegId(s.seg_id); setSelectedCandIdx(0); seekTo(s,0)} }>
                <td>{s.seg_id}</td>
                <td>{s.clip.scene_seg_idx}</td>
                <td>{t}</td>
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
          {(candList||[]).slice(0,50).map((c:any,i:number)=>{
            const on = i===selectedCandIdx
            return <div key={i} className='candidate'
                        style={{borderColor:on?'#409eff':'#eee', background:on?'#f5fbff':'#fff'}}
                        onClick={()=>{ setSelectedCandIdx(i); seekTo(selectedRow,i) }}>
              <div style={{display:'flex',justifyContent:'space-between',marginBottom:4}}>
                <div>scene {c.scene_id} / idx {c.scene_seg_idx}</div>
                <div style={{fontWeight:600}}>{(c.score??0).toFixed?.(3) ?? c.score}</div>
              </div>
              <div style={{fontSize:12,opacity:.7}}>t {c.start?.toFixed?.(2) ?? c.start} - {c.end?.toFixed?.(2) ?? c.end}</div>
              <div style={{fontSize:12,opacity:.6, marginTop:2}}>src: {c.source || '-'}</div>
            </div>
          })}
          <div style={{display:'flex', gap:8}}>
            <button onClick={acceptSelected}>应用所选</button>
            <button onClick={()=>{ setSelectedCandIdx(0); seekTo(selectedRow,0) }}>选第一个</button>
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