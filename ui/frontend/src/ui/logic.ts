// ui/frontend/src/ui/logic.ts
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { AxiosError } from 'axios'
import { listCandidates as apiListCandidates } from '../api'

async function fetchJSON<T=any>(url: string): Promise<T> {
  const r = await fetch(url, { credentials: 'same-origin' })
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json() as Promise<T>
}

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

  type Range = { clipStart:number, clipEnd:number, movieStart:number, movieEnd:number }
  const rangeRef = useRef<Range | null>(null)

  const [isPlaying, setIsPlaying] = useState(false)
  const [loopCount, setLoopCount] = useState(0)

  // 片段偏移（相对于当前流起点 base 的 offset）
  const clipStartOffsetRef = useRef(0)
  const movieStartOffsetRef = useRef(0)

  // 最近一次设置 src 后的“冷却窗口”，避免初始化阶段的 seek(0) 干扰
  const suppressUntilRef = useRef<number>(0)
  // 循环触发节流
  const lastLoopAtRef = useRef<number>(0)
  const LOOP_THROTTLE_MS = 300
  const END_EPS = 0.08

  // 记录当前已加载的 chunk，用来命中复用
  const lastClipUrlRef = useRef<string | null>(null)
  const lastMovieUrlRef = useRef<string | null>(null)
  const lastClipBaseRef = useRef<number | null>(null)
  const lastClipEffRef  = useRef<number | null>(null)
  const lastMovBaseRef  = useRef<number | null>(null)
  const lastMovEffRef   = useRef<number | null>(null)

  const play = useCallback(async ()=>{
    try {
      await clipRef.current?.play()
      await movieRef.current?.play()
      setIsPlaying(true)
    } catch (e) {
      if (debug) console.warn('[ctrl] play() blocked?', e)
      setIsPlaying(false)
    }
  }, [clipRef, movieRef, debug])

  const pause = useCallback(()=>{
    clipRef.current?.pause(); movieRef.current?.pause(); setIsPlaying(false)
  }, [clipRef, movieRef])

  // 定位接口（服务端返回 base/effective/offset）
  const locate = async (kind:'clip'|'movie', t:number, d:number, pre:number, post:number)=>{
    const qs = new URLSearchParams({ kind, t:String(t), d:String(d), pre:String(pre), post:String(post) })
    const r = await fetch(`${backendBase}/video/locate?`+qs.toString())
    const j = await r.json()
    if (debug) console.log('[locate]', j)
    return j as { ok?:boolean, base:number, effective:number, offset:number }
  }

  const waitSeeked = (v: HTMLVideoElement, target: number, eps = 0.05, timeoutMs = 1200) => new Promise<void>((resolve) => {
    let done = false
    const clear = () => { v.removeEventListener('seeked', onSeeked); v.removeEventListener('timeupdate', onTime); v.removeEventListener('loadedmetadata', onMeta); if(!done){done=true; resolve()} }
    const onSeeked = () => { if (Math.abs(v.currentTime - target) <= eps) clear() }
    const onTime = () => { if (Math.abs(v.currentTime - target) <= eps) clear() }
    const onMeta = () => { try { v.currentTime = target } catch {}
    }
    v.addEventListener('seeked', onSeeked)
    v.addEventListener('timeupdate', onTime)
    v.addEventListener('loadedmetadata', onMeta)
    // 超时兜底
    setTimeout(()=> clear(), timeoutMs)
  })

  const ensureSeek = (v: HTMLVideoElement | null, t: number) => {
    if (!v) return
    const doSeek = () => {
      try { v.currentTime = Math.max(t, 0) } catch {}
      // 双保险：短延迟再设一次，防止部分浏览器将第一次 seek 吃掉
      setTimeout(() => { try { v.currentTime = Math.max(t, 0) } catch {} }, 50)
    }
    if (v.readyState >= 1) doSeek()
    else {
      const onMeta = () => { v.removeEventListener('loadedmetadata', onMeta); doSeek() }
      v.addEventListener('loadedmetadata', onMeta)
    }
  }

  const playPair = useCallback(async (clipStart:number, clipEnd:number, movieStart:number, movieEnd:number)=>{
    const clipLen = Math.max(0.01, clipEnd - clipStart)
    const movLen  = Math.max(0.01, movieEnd - movieStart)

    // 预/后滚动（clip 更紧，movie 更宽）
    const preC=0.2, postC=0.2, preM=0.8, postM=0.8

    // 同步定位（base/effective/offset）
    let lc, lm
    try {
      [lc, lm] = await Promise.all([
        locate('clip',  clipStart, clipLen, preC, postC),
        locate('movie', movieStart, movLen,  preM, postM),
      ])
    } catch (e) {
      // 退化策略：用原始 t/d
      lc = { base: clipStart, effective: clipLen, offset: 0 }
      lm = { base: movieStart, effective: movLen,  offset: 0 }
    }

    clipStartOffsetRef.current = Math.max(0, Number(lc.offset)||0)
    movieStartOffsetRef.current = Math.max(0, Number(lm.offset)||0)

    // 构造流 URL
    const clipUrl  = `${backendBase}/video/clip?t=${Number(lc.base).toFixed(3)}&d=${Number(lc.effective).toFixed(3)}&pre=${preC}&post=${postC}`
    const movieUrl = `${backendBase}/video/movie?t=${Number(lm.base).toFixed(3)}&d=${Number(lm.effective).toFixed(3)}&pre=${preM}&post=${postM}`

    // 暂时禁用复用，确保 d 改变后浏览器真正拉取新的片段（避免看到旧的 duration）
    const finalClipUrl = clipUrl
    const finalMovieUrl = movieUrl

    onSetSrc?.(finalClipUrl, finalMovieUrl)
    lastClipUrlRef.current = finalClipUrl
    lastMovieUrlRef.current = finalMovieUrl
    lastClipBaseRef.current = Number(lc.base)
    lastClipEffRef.current  = Number(lc.effective)
    lastMovBaseRef.current  = Number(lm.base)
    lastMovEffRef.current   = Number(lm.effective)

    // 设定逻辑播放范围，供结束判断与进度条使用
    rangeRef.current = { clipStart, clipEnd, movieStart, movieEnd }
    setLoopCount(0)

    // 强制暂停，避免边换源边播放引发的自动回到 0 的行为
    pause()

    // 初始 seek 到各自 offset（相对当前流从 0 开始）
    const clipOffset  = Math.max(0, clipStartOffsetRef.current)
    const movieOffset = Math.max(0, movieStartOffsetRef.current)
    ensureSeek(clipRef.current,  clipOffset)
    ensureSeek(movieRef.current, movieOffset)

    // 冷却 1.2s，忽略初始化阶段的 0-seek 噪声
    suppressUntilRef.current = Date.now() + 1200

    // 等待两端至少一次 seek 落到位（或超时兜底）后再播放
    try {
      if (clipRef.current)  await waitSeeked(clipRef.current,  clipOffset)
      if (movieRef.current) await waitSeeked(movieRef.current, movieOffset)
    } catch {}

    await play()
  }, [backendBase, clipRef, movieRef, onSetSrc, play, pause])

  // RAF 轮询：仅在播放时检查是否到段尾，按 offset 回绕，并做节流
  useEffect(()=>{
    let raf = 0
    const tick = ()=>{
      const now = Date.now()
      if (now < suppressUntilRef.current) { raf = requestAnimationFrame(tick); return }

      const cv = clipRef.current, mv = movieRef.current
      const r = rangeRef.current
      if (!cv || !mv || !r || !syncPlay || !isPlaying) { raf = requestAnimationFrame(tick); return }

      const clipLen = Math.max(0.01, r.clipEnd - r.clipStart)
      const movLen  = Math.max(0.01, r.movieEnd - r.movieStart)

      const cOff = clipStartOffsetRef.current
      const mOff = movieStartOffsetRef.current

      const cAtEnd = cv.currentTime >= cOff + clipLen - END_EPS
      const mAtEnd = mv.currentTime >= mOff + movLen  - END_EPS

      if (cAtEnd || mAtEnd) {
        if (now - lastLoopAtRef.current > LOOP_THROTTLE_MS) {
          lastLoopAtRef.current = now
          const next = loopCount + 1
          if (next <= Math.max(1, maxLoops)) {
            try { cv.currentTime = cOff } catch {}
            try { mv.currentTime = mOff } catch {}
            try { cv.play() } catch {}
            try { mv.play() } catch {}
            setLoopCount(next)
          } else {
            pause()
          }
        }
      }

      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return ()=> cancelAnimationFrame(raf)
  }, [clipRef, movieRef, isPlaying, syncPlay, maxLoops, loopCount, pause])

  // 相对 seek：把滑块值映射到 offset
  const seekClipRel = useCallback((tRel:number)=>{
    const v = clipRef.current; const r = rangeRef.current; if (!v || !r) return
    const clipLen = Math.max(0.01, r.clipEnd - r.clipStart)
    const ratio = Math.max(0, Math.min(1, clipLen>0 ? tRel/clipLen : 0))
    try { v.currentTime = clipStartOffsetRef.current + ratio * clipLen } catch {}
    if (syncPlay) {
      const mv = movieRef.current; if (!mv) return
      const movLen = Math.max(0.01, r.movieEnd - r.movieStart)
      try { mv.currentTime = movieStartOffsetRef.current + ratio * movLen } catch {}
    }
  }, [clipRef, movieRef, syncPlay])

  const seekMovieRel = useCallback((tRel:number)=>{
    const v = movieRef.current; const r = rangeRef.current; if (!v || !r) return
    const movLen = Math.max(0.01, r.movieEnd - r.movieStart)
    const ratio = Math.max(0, Math.min(1, movLen>0 ? tRel/movLen : 0))
    try { v.currentTime = movieStartOffsetRef.current + ratio * movLen } catch {}
    if (syncPlay) {
      const cv = clipRef.current; if (!cv) return
      const clipLen = Math.max(0.01, r.clipEnd - r.clipStart)
      try { cv.currentTime = clipStartOffsetRef.current + ratio * clipLen } catch {}
    }
  }, [clipRef, movieRef, syncPlay])

  return {
    isPlaying,
    loopCount,
    range: rangeRef.current,
    playPair,
    play,
    pause,
    seekClipRel,
    seekMovieRel,
  }
}