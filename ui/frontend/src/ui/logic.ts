// ui/frontend/src/ui/logic.ts
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { AxiosError } from 'axios'
import { listCandidates as apiListCandidates } from '../api'

// ========== Types ==========

export type Scene = { clip_scene_id:number, count:number, avg_conf:number, chain_len:number }

export type Candidate = {
  seg_id?: number
  scene_id?: number
  scene_seg_idx?: number
  start?: number
  end?: number
  score?: number
  faiss_id?: number
  movie_id?: string
  shot_id?: number
  source?: string // 'top' | 'explain' | 'applied' | 'matched' | ...
}

export type SegmentRow = {
  seg_id: number
  clip: { scene_seg_idx?: number, start?: number, end?: number, scene_id?: number }
  matched_orig_seg?: Candidate | null
  top_matches?: Candidate[]
}

// ========== Utils ==========

export function formatTime(seconds: number): string {
  if (typeof seconds !== 'number' || isNaN(seconds)) return '0:00:00.000'
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)
  const ms = Math.floor((seconds % 1) * 1000)
  return `${hours}:${String(minutes).padStart(2,'0')}:${String(secs).padStart(2,'0')}.${String(ms).padStart(3,'0')}`
}

// 简单的 localStorage 状态——替换你现在在 App.tsx 里分散的 get/set
export function usePersistedState<T>(key: string, initial: T){
  const [val, setVal] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key)
      return raw ? (JSON.parse(raw) as T) : initial
    } catch { return initial }
  })
  useEffect(()=>{
    try { localStorage.setItem(key, JSON.stringify(val)) } catch {}
  }, [key, val])
  return [val, setVal] as const
}

// ========== Candidates：缓存 + 竞态保护 ==========

type Mode = 'top'|'scene'|'all'
type CacheKey = string

const candCache: Map<CacheKey, Candidate[]> = new Map()

export function useCandidates(segId: number | null, mode: Mode, k = 120){
  const [items, setItems] = useState<Candidate[]>([])
  const [total, setTotal] = useState<number>(0)
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const versionRef = useRef(0)

  const key = useMemo<CacheKey>(() => `${segId ?? 'nil'}:${mode}:${k}`, [segId, mode, k])

  const load = useCallback(async ()=>{
    if (segId == null) return
    // 先读缓存
    const cached = candCache.get(key)
    if (cached) {
      setItems(cached); setTotal(cached.length)
    }
    // 发起请求（带版本号，避免竞态）
    const myVersion = ++versionRef.current
    setLoading(true); setError(null)
    try {
      const resp = await apiListCandidates(segId, mode, k, 0)
      if (versionRef.current !== myVersion) return // 已过期
      const arr = (resp?.items || []) as Candidate[]
      candCache.set(key, arr)
      setItems(arr); setTotal(resp?.total ?? arr.length)
    } catch (e: unknown) {
      if (versionRef.current !== myVersion) return
      const err = (e as AxiosError)?.message || 'candidates failed'
      setError(err)
    } finally {
      if (versionRef.current === myVersion) setLoading(false)
    }
  }, [segId, mode, k, key])

  useEffect(()=>{ load() }, [load])

  return { items, total, loading, error, refresh: load }
}

// ========== Player Controller：双播放器控制统一入口 ==========

type PlayerOpts = {
  clipRef: React.RefObject<HTMLVideoElement>
  movieRef: React.RefObject<HTMLVideoElement>
  backendBase: string  // e.g. http://localhost:8787
  syncPlay: boolean
  maxLoops: number
  mirrorClip?: boolean
  debug?: boolean
  onSetSrc?: (clipUrl: string, movieUrl: string)=>void // 让 App.tsx 里去 setClipSrc/setMovieSrc
}

export function usePlayerController(opts: PlayerOpts){
  const { clipRef, movieRef, backendBase, syncPlay, maxLoops, mirrorClip, debug, onSetSrc } = opts
  const [isPlaying, setIsPlaying] = useState(false)
  const [loopCount, setLoopCount] = useState(0)
  const [range, setRange] = useState<{clipStart:number,clipEnd:number,movieStart:number,movieEnd:number} | null>(null)

  const clipHandlerRef = useRef<(this: HTMLVideoElement, ev: Event)=>any | null>(null)
  const movieHandlerRef = useRef<(this: HTMLVideoElement, ev: Event)=>any | null>(null)

  // 清理事件
  const detach = useCallback(()=>{
    const cv = clipRef.current, mv = movieRef.current
    if (cv && clipHandlerRef.current) { cv.removeEventListener('timeupdate', clipHandlerRef.current as any); clipHandlerRef.current = null }
    if (mv && movieHandlerRef.current) { mv.removeEventListener('timeupdate', movieHandlerRef.current as any); movieHandlerRef.current = null }
  }, [clipRef, movieRef])

  useEffect(()=>()=>detach(), [detach])

  const seekWhenReady = (v: HTMLVideoElement | null, t: number)=>{
    if (!v) return
    const doSeek = ()=>{ try { v.currentTime = Math.max(t, 0) } catch {} }
    if (v.readyState >= 1) doSeek()
    else {
      const onMeta = ()=>{ v.removeEventListener('loadedmetadata', onMeta); doSeek() }
      v.addEventListener('loadedmetadata', onMeta)
    }
  }

  const attachLoop = (v: HTMLVideoElement | null, s: number, e: number, side: 'clip'|'movie')=>{
    if (!v) return
    const handler = ()=>{
      if (!range) return
      if (e - s > 0.05 && v.currentTime > e - 0.03){
        // 到段尾：循环上限控制
        setLoopCount(prev=>{
          const next = prev + 1
          if (next >= maxLoops){
            try { v.pause() } catch {}
            if (syncPlay){ try { (side==='clip' ? movieRef.current : clipRef.current)?.pause() } catch {} }
            return next
          }
          try { v.currentTime = s } catch {}
          return next
        })
      }
    }
    v.addEventListener('timeupdate', handler)
    return handler
  }

  // 设置一对源并就位播放
  const playPair = useCallback((clipStart:number, clipEnd:number, movieStart:number, movieEnd:number)=>{
    const cv = clipRef.current, mv = movieRef.current
    setRange({ clipStart, clipEnd, movieStart, movieEnd })
    setLoopCount(0)
    // 构造后端流URL：?t=xxx
    const clipUrl = `${backendBase}/video/clip?t=${clipStart.toFixed(3)}`
    const movieUrl = `${backendBase}/video/movie?t=${movieStart.toFixed(3)}`
    // 交给外部设置 <video src>
    onSetSrc?.(clipUrl, movieUrl)

    // 由于是切片流，区间内 time 是从 0 开始
    seekWhenReady(cv, 0); seekWhenReady(mv, 0)

    // 重挂循环监听
    detach()
    clipHandlerRef.current = attachLoop(cv, 0, Math.max(0.01, clipEnd - clipStart), 'clip') || null
    movieHandlerRef.current = attachLoop(mv, 0, Math.max(0.01, movieEnd - movieStart), 'movie') || null

    // 播放
    const playWhenReady = async ()=>{
      const wait = (video: HTMLVideoElement)=> new Promise<void>(res=>{
        if (video.readyState >= 3) res()
        else {
          const onCan = ()=>{ video.removeEventListener('canplay', onCan); res() }
          video.addEventListener('canplay', onCan)
        }
      })
      try {
        if (cv) await wait(cv)
        if (mv) await wait(mv)
        await Promise.all([ cv?.play() || Promise.resolve(), mv?.play() || Promise.resolve() ])
        setIsPlaying(true)
      } catch {
        try { cv?.play() } catch {}
        try { mv?.play() } catch {}
        setIsPlaying(true)
      }
    }
    if (syncPlay) playWhenReady()
    else { try { cv?.play() } catch {}; try { mv?.play() } catch {}; setIsPlaying(true) }
  }, [backendBase, clipRef, movieRef, syncPlay, onSetSrc, detach])

  const play = useCallback(()=>{
    try { clipRef.current?.play() } catch {}
    try { movieRef.current?.play() } catch {}
    setIsPlaying(true)
  }, [clipRef, movieRef])

  const pause = useCallback(()=>{
    clipRef.current?.pause()
    movieRef.current?.pause()
    setIsPlaying(false)
  }, [clipRef, movieRef])

  // 相对区间 seek：tRel ∈ [0, clipRange] 或 [0, movieRange]
  const seekClipRel = useCallback((tRel: number)=>{
    const cv = clipRef.current
    if (!cv) return
    try { cv.currentTime = Math.max(0, tRel) } catch {}
    if (syncPlay){
      const mv = movieRef.current
      if (mv && range){
        const ratio = (cv.duration>0? tRel/cv.duration : 0)
        const t2 = ratio * (mv.duration || 0)
        try { mv.currentTime = t2 } catch {}
      }
    }
  }, [clipRef, movieRef, syncPlay, range])

  const seekMovieRel = useCallback((tRel: number)=>{
    const mv = movieRef.current
    if (!mv) return
    try { mv.currentTime = Math.max(0, tRel) } catch {}
    if (syncPlay){
      const cv = clipRef.current
      if (cv && range){
        const ratio = (mv.duration>0? tRel/mv.duration : 0)
        const t2 = ratio * (cv.duration || 0)
        try { cv.currentTime = t2 } catch {}
      }
    }
  }, [clipRef, movieRef, syncPlay, range])

  return {
    isPlaying,
    loopCount,
    range,
    playPair,
    play,
    pause,
    seekClipRel,
    seekMovieRel,
  }
}