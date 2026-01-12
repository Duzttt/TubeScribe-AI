/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./*.{js,ts,jsx,tsx}",           // <--- Added: Checks files in root (App.tsx, index.tsx)
    "./components/**/*.{js,ts,jsx,tsx}", // <--- Added: Checks files in components folder
    "./src/**/*.{js,ts,jsx,tsx}",    // (Optional: Keeps checking src if you create it later)
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        gray: {
          950: '#030712',
        }
      }
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}