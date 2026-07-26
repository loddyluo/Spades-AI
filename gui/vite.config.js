/*
 * Minimal Vite configuration for the local Spades table UI.
 * Input: none.
 * Output: a Vite config that serves the app from `gui/`.
 */
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 6006,
    allowedHosts: ['u867026-b009-6bd894a6.bjb2.seetacloud.com','uu867026-b009-6bd894a6.bjb2.seetacloud.com'],    // 允许所有主机访问（开发环境）
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8765',
        changeOrigin: true,
        ws: true,             // ← 关键：允许 WebSocket 升级
      },
    },
  },
});