export default {
  server: { port: 5173, proxy: { '/api': { target: 'http://127.0.0.1:8787', changeOrigin: true, rewrite: p=>p.replace(/^\/api/,'') } } }
}