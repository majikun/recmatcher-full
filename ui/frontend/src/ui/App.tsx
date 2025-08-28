import React, {useEffect, useState} from 'react'
import {openProject,listScenes,listSegments,applyChanges,save} from '../api'
type Scene = {clip_scene_id:number,count:number,avg_conf:number,chain_len:number}
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

  const [scenes,setScenes]=useState<Scene[]>([]); const [active,setActive]=useState<number|null>(null)
  const [segments,setSegments]=useState<any[]>([])
  async function doOpen(){
    await openProject(root, movie||undefined, clip||undefined)
    const arr: Scene[] = await listScenes()
    setScenes(arr)
    if (arr && arr.length) setActive(arr[0].clip_scene_id)
  }
  useEffect(()=>{ if(active!=null){ listSegments(active).then(setSegments) } },[active])
  return <div className='layout'>
    <div className='toolbar'>
      <input style={{width:320}} placeholder='project root' value={root} onChange={e=>{ const v=e.target.value; setRoot(v); try{ localStorage.setItem('rm_root', v); }catch{} }} />
      <input style={{width:260}} placeholder='movie.mp4 (optional)' value={movie} onChange={e=>{ const v=e.target.value; setMovie(v); try{ localStorage.setItem('rm_movie', v); }catch{} }} />
      <input style={{width:260}} placeholder='clip.mp4 (optional)' value={clip} onChange={e=>{ const v=e.target.value; setClip(v); try{ localStorage.setItem('rm_clip', v); }catch{} }} />
      <button onClick={doOpen}>打开</button><div style={{flex:1}}/>
      <button onClick={()=>save()}>保存导出</button>
    </div>
    <div className='main'>
      <div className='panel'>
        <div style={{fontWeight:600,marginBottom:6}}>Clip 场景</div>
        <div className='scene-list'>
          {Array.isArray(scenes) && scenes.map((s:Scene)=>(
            <div key={s.clip_scene_id}
                 className={'scene-item '+(active===s.clip_scene_id?'active':'')}
                 onClick={()=>setActive(s.clip_scene_id)}>
              <div>#{s.clip_scene_id}</div>
              <div className='badge'>{s.count} / len {s.chain_len}</div>
            </div>
          ))}
        </div>
      </div>
      <div className='panel'>
        <div className='videos'>
          <div><div style={{fontSize:12,opacity:.7,marginBottom:4}}>Clip</div><video src='/api/video/clip' controls/></div>
          <div><div style={{fontSize:12,opacity:.7,marginBottom:4}}>Movie</div><video src='/api/video/movie' controls/></div>
        </div>
        <table className='seg-table'><thead><tr><th>seg_id</th><th>clip idx</th><th>clip t</th><th>matched scene/idx</th><th>score</th><th>操作</th></tr></thead>
          <tbody>{segments.map((s:any)=>{ const mo=s.matched_orig_seg||{}; const t=`${(s.clip.start??0).toFixed(2)}-${(s.clip.end??0).toFixed(2)}`
            return <tr key={s.seg_id}><td>{s.seg_id}</td><td>{s.clip.scene_seg_idx}</td><td>{t}</td>
              <td>{mo.scene_id??'-'} / {mo.scene_seg_idx??'-'}</td><td>{(mo.score??0).toFixed(3)}</td>
              <td><button onClick={()=>{ const cand=(s.top_matches&&s.top_matches[0])||null; if(!cand) return;
                applyChanges([{seg_id:s.seg_id, chosen:cand}]).then(()=>listSegments(active!).then(setSegments))}}>接受候选</button></td>
            </tr>})}</tbody></table>
      </div>
      <div className='panel'><div style={{fontWeight:600,marginBottom:6}}>候选（当前段）</div><div style={{fontSize:12,opacity:.7}}>待实现</div></div>
    </div>
  </div>
}
