import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/llm_reliability_framework/', 
  build: {
    outDir: '../docs',
    emptyOutDir: true,
  }
})
