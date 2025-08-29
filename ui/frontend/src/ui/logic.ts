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
  sidecarBase?: string
}

export function usePlayerController(opts: PlayerOpts){
  const {
    clipRef,
    movieRef,
    backendBase,     // kept for backward compat (unused in sidecar mode)
    sidecarBase: sidecarBaseFromOpts,
    syncPlay,
    maxLoops,
    mirrorClip,
    debug,
    onSetSrc
  } = opts

  // ---- Sidecar base (default to localhost:9777) ----
  const SIDECAR_BASE = sidecarBaseFromOpts || `${location.protocol}//${location.hostname}:9777`
  const USE_SIDECAR = true  // we now always prefer sidecar

  // ---- Shared shape with old controller for App.tsx compatibility ----
  type Range = { clipStart:number, clipEnd:number, movieStart:number, movieEnd:number }
  const rangeRef = useRef<Range | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [loopCount, setLoopCount] = useState(0)

  // Keep offsets for progress-bar mapping (in sidecar mode these are logical 0)
  const clipStartOffsetRef = useRef(0)
  const movieStartOffsetRef = useRef(0)

  // Only used by legacy streaming path (kept for type parity)
  const suppressUntilRef = useRef<number>(0)

  // Latest command wins
  const actionIdRef = useRef(0)

  // ---------- helpers ----------
  const postJSON = async (url: string, body: any) => {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body ?? {})
    })
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    return r.json().catch(()=> ({}))
  }

  const getJSON = async (url: string) => {
    const r = await fetch(url)
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    return r.json()
  }

  // ---------- core controls (sidecar) ----------
  const play = useCallback(async ()=>{
    if (!USE_SIDECAR) {
      try { await clipRef.current?.play(); await movieRef.current?.play() } catch {}
      setIsPlaying(true)
      return
    }
    try {
      await postJSON(`${SIDECAR_BASE}/resume`, {})
      setIsPlaying(true)
    } catch (e) {
      if (debug) console.warn('[sidecar] resume failed', e)
      setIsPlaying(false)
    }
  }, [SIDECAR_BASE, USE_SIDECAR, clipRef, movieRef, debug])

  const pause = useCallback(async ()=>{
    if (!USE_SIDECAR) {
      clipRef.current?.pause(); movieRef.current?.pause()
      setIsPlaying(false)
      return
    }
    try {
      await postJSON(`${SIDECAR_BASE}/pause`, {})
    } catch (e) {
      if (debug) console.warn('[sidecar] pause failed', e)
    } finally {
      setIsPlaying(false)
    }
  }, [SIDECAR_BASE, USE_SIDECAR, clipRef, movieRef, debug])

  const playPair = useCallback(async (clipStart:number, clipEnd:number, movieStart:number, movieEnd:number)=>{
    const myId = ++actionIdRef.current

    // define logical range for UI display / loop logic
    rangeRef.current = { clipStart, clipEnd, movieStart, movieEnd }
    setLoopCount(0)

    // In sidecar mode we do not set video src at all
    if (onSetSrc) onSetSrc('', '')

    if (!USE_SIDECAR) {
      // legacy path (kept only for parity; not used)
      try {
        clipRef.current && (clipRef.current.currentTime = 0)
        movieRef.current && (movieRef.current.currentTime = 0)
      } catch {}
      setIsPlaying(true)
      return
    }

    // Call sidecar to start pair playback (AB loop is handled by sidecar)
    try {
      const payload = {
        clipStart, clipEnd,
        movieStart, movieEnd,
        loops: Math.max(1, maxLoops),
        sync: !!syncPlay
      }
      if (debug) console.log('[sidecar] /play_pair', payload)
      await postJSON(`${SIDECAR_BASE}/play_pair`, payload)
      if (myId !== actionIdRef.current) return
      setIsPlaying(true)
    } catch (e) {
      console.error('[sidecar] play_pair failed', e)
      setIsPlaying(false)
    }
  }, [SIDECAR_BASE, USE_SIDECAR, maxLoops, syncPlay, onSetSrc, clipRef, movieRef, debug])

  // Relative seek (map UI slider 0..len to absolute inside current range)
  const seekClipRel = useCallback(async (tRel:number)=>{
    const r = rangeRef.current; if (!r) return
    const len = Math.max(0.01, r.clipEnd - r.clipStart)
    const ratio = Math.max(0, Math.min(1, len>0 ? tRel/len : 0))
    if (!USE_SIDECAR) {
      try { if (clipRef.current) clipRef.current.currentTime = ratio * len } catch {}
      if (syncPlay) try { if (movieRef.current) movieRef.current.currentTime = ratio * Math.max(0.01, r.movieEnd - r.movieStart) } catch {}
      return
    }
    try {
      await postJSON(`${SIDECAR_BASE}/seek_rel`, { ratio })
    } catch (e) {
      if (debug) console.warn('[sidecar] seek_rel failed', e)
    }
  }, [SIDECAR_BASE, USE_SIDECAR, clipRef, movieRef, syncPlay, debug])

  const seekMovieRel = useCallback(async (tRel:number)=>{
    // We keep the same behaviour as seekClipRel: single ratio controls both when syncPlay=true
    return seekClipRel(tRel)
  }, [seekClipRel])

  // ---------- poll sidecar status for loop count / play state ----------
  useEffect(()=>{
    if (!USE_SIDECAR) return
    let timer = 0
    const tick = async ()=>{
      try {
        const s = await getJSON(`${SIDECAR_BASE}/status`)
        // Expected: { playing: boolean, loopCount?: number }
        if (typeof s?.playing === 'boolean') setIsPlaying(!!s.playing)
        if (typeof s?.loopCount === 'number') setLoopCount(s.loopCount)
      } catch (e) {
        // ignore
      } finally {
        timer = window.setTimeout(tick, 250)
      }
    }
    timer = window.setTimeout(tick, 250)
    return ()=> { if (timer) clearTimeout(timer) }
  }, [SIDECAR_BASE, USE_SIDECAR])

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