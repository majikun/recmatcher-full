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
  const HARD_CLAMP_EPS = 0.15

  // 记录当前已加载的 chunk，用来命中复用
  const lastClipUrlRef = useRef<string | null>(null)
  const lastMovieUrlRef = useRef<string | null>(null)
  const lastClipBaseRef = useRef<number | null>(null)
  const lastClipEffRef  = useRef<number | null>(null)
  const lastMovBaseRef  = useRef<number | null>(null)
  const lastMovEffRef   = useRef<number | null>(null)

  // 轻量定位缓存（降低同一段短时间内重复 locate 带来的抖动）
  const locateCacheRef = useRef<Map<string, {base:number, effective:number, offset:number, ts:number}>>(new Map())
  const LOCATE_TTL_MS = 2000

  // 并发保护：只让最新的一次 playPair 生效
  const actionIdRef = useRef(0)

  const locate = async (kind:'clip'|'movie', t:number, d:number, pre:number, post:number)=>{
    // 归一化 key，避免浮点毛刺导致缓存失效
    const key = `${kind}:${t.toFixed(3)}:${d.toFixed(3)}:${pre}:${post}`
    const now = Date.now()
    const cache = locateCacheRef.current
    const hit = cache.get(key)
    if (hit && (now - hit.ts) < LOCATE_TTL_MS){
      if (debug) console.log('[locate:cache]', key, hit)
      return { base: hit.base, effective: hit.effective, offset: hit.offset }
    }
    const qs = new URLSearchParams({ kind, t:String(t), d:String(d), pre:String(pre), post:String(post) })
    const r = await fetch(`${backendBase}/video/locate?`+qs.toString())
    const j = await r.json()
    if (debug) console.log('[locate]', j)
    const rec = { base: Number(j.base)||t, effective: Number(j.effective)||d, offset: Number(j.offset)||0, ts: now }
    cache.set(key, rec)
    return { base: rec.base, effective: rec.effective, offset: rec.offset }
  }

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
    const myId = ++actionIdRef.current

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
    } catch {
      lc = { base: clipStart, effective: clipLen, offset: 0 }
      lm = { base: movieStart, effective: movLen,  offset: 0 }
    }

    // 如果期间有更新的 playPair 抢先了，放弃本次
    if (myId !== actionIdRef.current) return

    clipStartOffsetRef.current = Math.max(0, Number(lc.offset)||0)
    movieStartOffsetRef.current = Math.max(0, Number(lm.offset)||0)

    // 构造流 URL
    const clipUrl  = `${backendBase}/video/clip?t=${Number(lc.base).toFixed(3)}&d=${Number(lc.effective).toFixed(3)}&pre=${preC}&post=${postC}`
    const movieUrl = `${backendBase}/video/movie?t=${Number(lm.base).toFixed(3)}&d=${Number(lm.effective).toFixed(3)}&pre=${preM}&post=${postM}`

    const needSetClip  = (clipUrl  !== lastClipUrlRef.current)
    const needSetMovie = (movieUrl !== lastMovieUrlRef.current)

    // 只有在 URL 变化时才更新 src，避免重复解码引起的抖动
    if (needSetClip || needSetMovie){
      onSetSrc?.(needSetClip ? clipUrl : (lastClipUrlRef.current || clipUrl),
                 needSetMovie ? movieUrl : (lastMovieUrlRef.current || movieUrl))
      lastClipUrlRef.current  = clipUrl
      lastMovieUrlRef.current = movieUrl
      lastClipBaseRef.current = Number(lc.base)
      lastClipEffRef.current  = Number(lc.effective)
      lastMovBaseRef.current  = Number(lm.base)
      lastMovEffRef.current   = Number(lm.effective)
      // 冷却：仅在真正换源时启用
      suppressUntilRef.current = Date.now() + 1200
    }

    // 设定逻辑播放范围，供结束判断与进度条使用
    rangeRef.current = { clipStart, clipEnd, movieStart, movieEnd }
    setLoopCount(0)

    // 初始 seek 到各自 offset（相对当前流从 0 开始）
    const clipOffset  = Math.max(0, clipStartOffsetRef.current)
    const movieOffset = Math.max(0, movieStartOffsetRef.current)
    ensureSeek(clipRef.current,  clipOffset)
    ensureSeek(movieRef.current, movieOffset)

    // 等待就位（只要 latest）
    if (myId !== actionIdRef.current) return
    try {
      if (clipRef.current)  await waitSeeked(clipRef.current,  clipOffset)
      if (movieRef.current) await waitSeeked(movieRef.current, movieOffset)
    } catch {}

    if (myId !== actionIdRef.current) return
    await play()
  }, [backendBase, clipRef, movieRef, onSetSrc, play, pause])

  useEffect(()=>{
    const cv = clipRef.current
    const mv = movieRef.current
    if (!cv || !mv) return

    const handleEnded = () => {
      const r = rangeRef.current
      if (!r) return
      const cOff = clipStartOffsetRef.current
      const mOff = movieStartOffsetRef.current
      const clipLen = Math.max(0.01, r.clipEnd - r.clipStart)
      const movLen  = Math.max(0.01, r.movieEnd - r.movieStart)

      const next = loopCount + 1
      const allowLoop = Math.max(1, maxLoops) > 1

      if (allowLoop && next <= Math.max(1, maxLoops)){
        try { cv.currentTime = cOff } catch {}
        try { mv.currentTime = mOff } catch {}
        try { cv.play() } catch {}
        try { mv.play() } catch {}
        setLoopCount(next)
      } else {
        try { cv.currentTime = Math.min(cv.currentTime, cOff + clipLen) } catch {}
        try { mv.currentTime = Math.min(mv.currentTime, mOff + movLen) } catch {}
        pause()
      }
    }

    cv.addEventListener('ended', handleEnded)
    mv.addEventListener('ended', handleEnded)
    return ()=>{
      cv.removeEventListener('ended', handleEnded)
      mv.removeEventListener('ended', handleEnded)
    }
  }, [clipRef, movieRef, maxLoops, loopCount, pause])

  // RAF 轮询：仅在播放时检查是否到段尾，按 offset 回绕，并做节流
  useEffect(()=>{
    let raf = 0
    const tick = ()=>{
      const now = Date.now()
      if (now < suppressUntilRef.current) { raf = requestAnimationFrame(tick); return }

      const cv = clipRef.current, mv = movieRef.current
      const r = rangeRef.current
      if (!cv || !mv || !r || !isPlaying) { raf = requestAnimationFrame(tick); return }

      const clipLen = Math.max(0.01, r.clipEnd - r.clipStart)
      const movLen  = Math.max(0.01, r.movieEnd - r.movieStart)

      const cOff = clipStartOffsetRef.current
      const mOff = movieStartOffsetRef.current

      const cAtEnd = cv.currentTime >= cOff + clipLen - END_EPS
      const mAtEnd = mv.currentTime >= mOff + movLen  - END_EPS

      const overEndHard = (cv.currentTime > cOff + clipLen + HARD_CLAMP_EPS) ||
                          (mv.currentTime > mOff + movLen  + HARD_CLAMP_EPS)

      if (cAtEnd || mAtEnd || overEndHard) {
        if (now - lastLoopAtRef.current > LOOP_THROTTLE_MS) {
          lastLoopAtRef.current = now
          const next = loopCount + 1
          const allowLoop = Math.max(1, maxLoops) > 1

          if (allowLoop && next <= Math.max(1, maxLoops)) {
            // 回到 offset，从头循环
            try { cv.currentTime = cOff } catch {}
            try { mv.currentTime = mOff } catch {}
            try { cv.play() } catch {}
            try { mv.play() } catch {}
            setLoopCount(next)
          } else {
            // 不循环：精确夹到尾点再暂停，避免继续播放到补充片段
            try { cv.currentTime = Math.min(cv.currentTime, cOff + clipLen) } catch {}
            try { mv.currentTime = Math.min(mv.currentTime, mOff + movLen) } catch {}
            pause()
          }
        }
      }

      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return ()=> cancelAnimationFrame(raf)
  }, [clipRef, movieRef, isPlaying, maxLoops, loopCount, pause])

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
    getOffsets: () => ({ clipOffset: clipStartOffsetRef.current, movieOffset: movieStartOffsetRef.current }),
  }
}