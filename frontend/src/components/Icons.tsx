/**
 * 图标集：内联 SVG，不引入图标库（保持依赖精简、可按 currentColor 着色）。
 * 线条粗细统一 1.75，风格贴近 DeepSeek 官网的细线极简风。
 */

import type { SVGProps } from 'react'

type IconProps = SVGProps<SVGSVGElement> & { size?: number }

function Svg({ size = 18, children, ...rest }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {children}
    </svg>
  )
}

export const IconPlus = (p: IconProps) => (
  <Svg {...p}>
    <path d="M12 5v14M5 12h14" />
  </Svg>
)

export const IconPanel = (p: IconProps) => (
  <Svg {...p}>
    <rect x="3" y="4" width="18" height="16" rx="2.5" />
    <path d="M9 4v16" />
  </Svg>
)

export const IconMenu = (p: IconProps) => (
  <Svg {...p}>
    <path d="M4 7h16M4 12h16M4 17h16" />
  </Svg>
)

export const IconSearch = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="11" cy="11" r="7" />
    <path d="m20 20-3.2-3.2" />
  </Svg>
)

export const IconEdit = (p: IconProps) => (
  <Svg {...p}>
    <path d="M12 20h9" />
    <path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z" />
  </Svg>
)

export const IconTrash = (p: IconProps) => (
  <Svg {...p}>
    <path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14" />
    <path d="M10 11v6M14 11v6" />
  </Svg>
)

export const IconCopy = (p: IconProps) => (
  <Svg {...p}>
    <rect x="9" y="9" width="11" height="11" rx="2" />
    <path d="M5 15V5a2 2 0 0 1 2-2h8" />
  </Svg>
)

export const IconCheck = (p: IconProps) => (
  <Svg {...p}>
    <path d="m4 12.5 5 5L20 7" />
  </Svg>
)

export const IconThumbUp = (p: IconProps) => (
  <Svg {...p}>
    <path d="M7 22V11l4.5-8a2 2 0 0 1 2.9 2.4L13 10h4.8a2.2 2.2 0 0 1 2.1 2.8l-1.8 6.6A2.2 2.2 0 0 1 16 21H7Z" />
    <path d="M7 11H4v11h3" />
  </Svg>
)

export const IconThumbDown = (p: IconProps) => (
  <Svg {...p}>
    <path d="M17 2v11l-4.5 8a2 2 0 0 1-2.9-2.4L11 14H6.2a2.2 2.2 0 0 1-2.1-2.8l1.8-6.6A2.2 2.2 0 0 1 8 3h9Z" />
    <path d="M17 13h3V2h-3" />
  </Svg>
)

export const IconRefresh = (p: IconProps) => (
  <Svg {...p}>
    <path d="M20 11a8 8 0 0 0-13.7-5.3L3 9" />
    <path d="M4 13a8 8 0 0 0 13.7 5.3L21 15" />
    <path d="M3 4v5h5M21 20v-5h-5" />
  </Svg>
)

export const IconSend = (p: IconProps) => (
  <Svg {...p}>
    <path d="M12 19V5" />
    <path d="m5.5 11.5 6.5-6.5 6.5 6.5" />
  </Svg>
)

export const IconStop = (p: IconProps) => (
  <Svg {...p} strokeWidth={0}>
    <rect x="6.5" y="6.5" width="11" height="11" rx="2" fill="currentColor" />
  </Svg>
)

export const IconPaperclip = (p: IconProps) => (
  <Svg {...p}>
    <path d="M21 11.5 12.5 20a5 5 0 0 1-7-7l8-8a3.5 3.5 0 0 1 5 5l-8 8a2 2 0 0 1-3-3l7.5-7.5" />
  </Svg>
)

export const IconSun = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
  </Svg>
)

export const IconMoon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5Z" />
  </Svg>
)

export const IconX = (p: IconProps) => (
  <Svg {...p}>
    <path d="M6 6l12 12M18 6L6 18" />
  </Svg>
)

export const IconFile = (p: IconProps) => (
  <Svg {...p}>
    <path d="M14 3v5h5" />
    <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h9l5 5v11a2 2 0 0 1-2 2Z" />
  </Svg>
)

export const IconDoc = (p: IconProps) => (
  <Svg {...p}>
    <path d="M14 3v5h5" />
    <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h9l5 5v11a2 2 0 0 1-2 2Z" />
    <path d="M8 13h8M8 17h5" />
  </Svg>
)

export const IconAlert = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 8v5M12 16.5v.5" />
  </Svg>
)

export const IconSparkle = (p: IconProps) => (
  <Svg {...p}>
    <path d="M12 3l1.8 4.9L18.7 9.7l-4.9 1.8L12 16.4l-1.8-4.9L5.3 9.7l4.9-1.8L12 3Z" />
    <path d="M18.5 16.5l.8 2 2 .8-2 .8-.8 2-.8-2-2-.8 2-.8.8-2Z" />
  </Svg>
)

export const IconChevron = (p: IconProps) => (
  <Svg {...p}>
    <path d="m6 9 6 6 6-6" />
  </Svg>
)

export const IconBrain = (p: IconProps) => (
  <Svg {...p}>
    <path d="M9.5 4A2.5 2.5 0 0 0 7 6.5 2.5 2.5 0 0 0 4.5 9 2.5 2.5 0 0 0 5 13.7 2.5 2.5 0 0 0 7 18a2.5 2.5 0 0 0 4.5 1.5V4.9A2.5 2.5 0 0 0 9.5 4Z" />
    <path d="M14.5 4A2.5 2.5 0 0 1 17 6.5 2.5 2.5 0 0 1 19.5 9a2.5 2.5 0 0 1-.5 4.7A2.5 2.5 0 0 1 17 18a2.5 2.5 0 0 1-4.5 1.5" />
  </Svg>
)

export const IconDatabase = (p: IconProps) => (
  <Svg {...p}>
    <ellipse cx="12" cy="6" rx="7" ry="3" />
    <path d="M5 6v12c0 1.7 3.1 3 7 3s7-1.3 7-3V6" />
    <path d="M5 12c0 1.7 3.1 3 7 3s7-1.3 7-3" />
  </Svg>
)

export const IconLink = (p: IconProps) => (
  <Svg {...p}>
    <path d="M10 13a4 4 0 0 0 5.7 0l2.6-2.6a4 4 0 0 0-5.7-5.7L11.4 6" />
    <path d="M14 11a4 4 0 0 0-5.7 0L5.7 13.6a4 4 0 0 0 5.7 5.7L12.6 18" />
  </Svg>
)

/** DeepSeek 风格的四叶草 Logo（品牌标记，非官方素材）。 */
export const Logo = ({ size = 32, className }: { size?: number; className?: string }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 48 48"
    className={className}
    aria-hidden="true"
    focusable="false"
  >
    <defs>
      <linearGradient id="logo-grad" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stopColor="#5B8DEF" />
        <stop offset="100%" stopColor="#4D6BFE" />
      </linearGradient>
    </defs>
    <rect width="48" height="48" rx="12" fill="url(#logo-grad)" />
    <path
      d="M24 11c1.7 5 3.9 7.2 8.9 8.9-5 1.7-7.2 3.9-8.9 8.9-1.7-5-3.9-7.2-8.9-8.9 5-1.7 7.2-3.9 8.9-8.9Z"
      fill="#fff"
      opacity="0.96"
    />
    <path
      d="M31.5 28.5c.85 2.5 1.95 3.6 4.45 4.45-2.5.85-3.6 1.95-4.45 4.45-.85-2.5-1.95-3.6-4.45-4.45 2.5-.85 3.6-1.95 4.45-4.45Z"
      fill="#fff"
      opacity="0.72"
    />
  </svg>
)
