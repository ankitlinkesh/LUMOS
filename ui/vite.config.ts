import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build to static files FastAPI serves (triad/api/app.py mounts ui/dist at "/").
// Relative base so assets resolve correctly when served from "/".
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
});
