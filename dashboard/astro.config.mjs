import { defineConfig } from 'astro/config';
import react from '@astrojs/react';

export default defineConfig({
  integrations: [react()],
  server: { host: '0.0.0.0', port: 4321 },
  vite: {
    server: {
      allowedHosts: ['paul-cheddar', 'pauls-macbook-pro'],
      proxy: {
        '/api': {
          target: 'http://localhost:6060',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ''),
        },
      },
    },
  },
});
