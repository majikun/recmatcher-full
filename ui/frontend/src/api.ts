import axios from 'axios'
export const api = axios.create({ baseURL: '/api' })
export const openProject = async (root:string, movie?:string, clip?:string) =>
  (await api.post('/project/open', { root, movie_path: movie, clip_path: clip })).data
export const listScenes = async () => (await api.get('/scenes')).data
export const listSegments = async (clip_scene_id:number) => (await api.get('/segments',{params:{clip_scene_id}})).data
export const applyChanges = async (changes:any[]) => (await api.post('/apply',{changes})).data
export const save = async (out_path?:string) => (await api.post('/save',{out_path})).data
