import type { Config } from 'tailwindcss'

/**
 * 设计令牌统一走 CSS 变量（见 src/styles/index.css），
 * 这样深/浅色切换只需给 <html> 加减一个 class，无需到处写 dark: 前缀。
 */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // 主色调：低饱和度冰淇淋蓝（DeepSeek 品牌色）
        brand: {
          DEFAULT: '#4D6BFE',
          hover: '#3D5AF0',
          active: '#3548D4',
          soft: 'rgb(var(--brand-soft) / <alpha-value>)',
          fg: '#FFFFFF',
        },
        // 语义化表面与文字，全部指向 CSS 变量
        canvas: 'rgb(var(--canvas) / <alpha-value>)',
        surface: 'rgb(var(--surface) / <alpha-value>)',
        elevated: 'rgb(var(--elevated) / <alpha-value>)',
        hover: 'rgb(var(--hover) / <alpha-value>)',
        line: 'rgb(var(--line) / <alpha-value>)',
        ink: 'rgb(var(--ink) / <alpha-value>)',
        'ink-soft': 'rgb(var(--ink-soft) / <alpha-value>)',
        'ink-muted': 'rgb(var(--ink-muted) / <alpha-value>)',
        // 用户气泡
        bubble: 'rgb(var(--bubble) / <alpha-value>)',
        danger: '#F04438',
        success: '#12B76A',
      },
      fontFamily: {
        sans: [
          '-apple-system',
          'BlinkMacSystemFont',
          'Inter',
          '"Segoe UI"',
          '"PingFang SC"',
          '"Hiragino Sans GB"',
          '"Microsoft YaHei"',
          'sans-serif',
        ],
        mono: [
          'ui-monospace',
          'SFMono-Regular',
          'Menlo',
          'Consolas',
          '"Liberation Mono"',
          'monospace',
        ],
      },
      borderRadius: {
        // 大圆角风格
        md: '10px',
        lg: '12px',
        xl: '16px',
        '2xl': '20px',
        '3xl': '24px',
      },
      spacing: {
        sidebar: '260px',
      },
      maxWidth: {
        // 对话正文的最大宽度：太宽会降低可读性
        chat: '768px',
      },
      boxShadow: {
        card: '0 1px 3px rgb(0 0 0 / 0.06), 0 1px 2px rgb(0 0 0 / 0.04)',
        pop: '0 8px 24px rgb(0 0 0 / 0.12)',
        composer: '0 2px 12px rgb(0 0 0 / 0.06)',
      },
      keyframes: {
        'fade-in': {
          from: { opacity: '0', transform: 'translateY(4px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        'pulse-dot': {
          '0%, 80%, 100%': { opacity: '0.25', transform: 'scale(0.85)' },
          '40%': { opacity: '1', transform: 'scale(1)' },
        },
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0' },
        },
      },
      animation: {
        'fade-in': 'fade-in 0.18s ease-out',
        'pulse-dot': 'pulse-dot 1.2s infinite ease-in-out',
        blink: 'blink 1s step-end infinite',
      },
      transitionTimingFunction: {
        smooth: 'cubic-bezier(0.4, 0, 0.2, 1)',
      },
    },
  },
  plugins: [],
} satisfies Config
