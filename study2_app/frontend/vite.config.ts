import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Override with PREFQUEST_BACKEND_URL env var if backend is on a non-default port/host.
const BACKEND = process.env.PREFQUEST_BACKEND_URL ?? 'http://localhost:8001'

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
