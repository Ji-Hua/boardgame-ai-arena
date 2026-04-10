import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 8765,
    strictPort: true,
  },
  preview: {
    host: '0.0.0.0',
    port: 8765,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://backend:8764',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://backend:8764',
        ws: true,
        changeOrigin: true,
      },
    },
  },
})
