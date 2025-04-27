import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Define Vite configuration
export default defineConfig({
  plugins: [react()],
  base: '/', // Important for correct routing of assets
  build: {
    outDir: 'dist',  // Ensure this matches the directory in your Azure deployment
  },
})