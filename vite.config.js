import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // If you deploy under a sub-path (e.g. GitHub Pages: username.github.io/chemometrics-lab/),
  // set base to "/chemometrics-lab/". For a root domain or Vercel/Netlify, leave as "/".
  base: "/",
});
