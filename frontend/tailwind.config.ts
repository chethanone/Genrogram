import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        background: '#F4F1EE',
        surface: '#FFFFFF',
        'dark-surface': '#171717',
        primary: '#161616',
        secondary: '#77716E',
        accent: {
          DEFAULT: '#C96F70',
          hover: '#B55D5E',
          subtle: 'rgba(201, 111, 112, 0.12)',
        },
        border: 'rgba(22, 22, 22, 0.1)',
      },
      fontFamily: {
        sans: ['Space Grotesk', 'sans-serif'],
        mono: ['DM Mono', 'monospace'],
      },
    },
  },
  plugins: [],
};

export default config;
