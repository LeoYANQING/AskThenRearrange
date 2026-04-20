import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const BACKEND = 'http://localhost:8001'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/sessions': BACKEND,
      '/dialogue': BACKEND,
      '/logs': BACKEND,
      '/health': BACKEND,
      '/voice': BACKEND,
    },
  },
})
