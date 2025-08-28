import axios from 'axios'

export const api = axios.create({
  baseURL: '/api',
})

export async function openProject(root: string, movie_path?: string, clip_path?: string) {
  const res = await api.post('/project/open', { root, movie_path, clip_path })
  return res.data
}

export async function listScenes() {
  const res = await api.get('/scenes')
  const data = res.data
  const arr = Array.isArray(data) ? data : (data?.scenes ?? [])
  return arr as { clip_scene_id: number, count: number, avg_conf: number, chain_len: number }[]
}

export async function listSegments(clip_scene_id: number) {
  const res = await api.get('/segments', { params: { clip_scene_id } })
  // backend already returns an array; add a guard anyway
  const data = res.data
  return Array.isArray(data) ? data : (data?.segments ?? [])
}

export async function applyChanges(changes: any[]) {
  const res = await api.post('/apply', { changes })
  return res.data
}

export async function rebuildScene(clip_scene_id: number) {
  const res = await api.post('/ops/rebuild_scene', { clip_scene_id })
  return res.data
}

export async function save(out_path?: string) {
  const res = await api.post('/save', { out_path })
  return res.data
}