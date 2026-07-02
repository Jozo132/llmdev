/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{vue,ts}"],
  theme: {
    extend: {
      colors: {
        canvas: "#0b0f19",
        panel: "#111827",
        node: {
          data: "#0ea5e9",
          tokenizer: "#8b5cf6",
          model: "#f59e0b",
          train: "#ef4444",
          eval: "#10b981",
          export: "#64748b",
          custom: "#ec4899",
        },
      },
    },
  },
  plugins: [],
};
