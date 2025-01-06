import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // This tells Vite to proxy any request starting with /api to your FastAPI server
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        configure: (proxy, options) => {
          // Log all proxy events
          proxy.on('error', (err, req, res) => {
            console.log('=== Proxy Error ===');
            console.log('Error:', err);
            console.log('Request URL:', req.url);
            console.log('=== End Error ===\n');
          });

          proxy.on('proxyReq', (proxyReq, req, res) => {
            console.log('=== Proxy Request ===');
            console.log('Original URL:', req.url);
            console.log('Proxy URL:', proxyReq.path);
            console.log('Target:', options.target);
            console.log('=== End Proxy Request ===\n');
          });

          proxy.on('proxyRes', (proxyRes, req, res) => {
            console.log('=== Proxy Response ===');
            console.log('Status:', proxyRes.statusCode);
            console.log('Headers:', proxyRes.headers);
            console.log('=== End Proxy Response ===\n');
          });
        }
      }
    }
  }
})
