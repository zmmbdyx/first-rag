/** 时间与文案小工具。 */

/** 把 ISO 时间串安全转成 Date（后端返回的是 UTC，无时区后缀时补 Z）。 */
export function parseTime(value: string | null | undefined): Date | null {
  if (!value) return null
  // FastAPI 序列化 datetime 时可能不带时区（"2026-01-01T12:00:00"），
  // 统一按 UTC 解析，避免时区偏移导致「今天/昨天」分组错位。
  const normalized = /[zZ]|[+-]\d{2}:?\d{2}$/.test(value) ? value : `${value}Z`
  const d = new Date(normalized)
  return Number.isNaN(d.getTime()) ? null : d
}

/** 当天 0 点。 */
function startOfDay(d: Date): number {
  return new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime()
}

export type TimeGroup = '今天' | '昨天' | '近7天' | '更早'

/** 按更新时间把会话分组（与侧边栏的分组标题一致）。 */
export function timeGroupOf(value: string | null | undefined): TimeGroup {
  const d = parseTime(value)
  if (!d) return '更早'
  const today = startOfDay(new Date())
  const day = startOfDay(d)
  const diffDays = Math.round((today - day) / 86_400_000)
  if (diffDays <= 0) return '今天'
  if (diffDays === 1) return '昨天'
  if (diffDays < 7) return '近7天'
  return '更早'
}

/** 相对时间，如「刚刚」「3 分钟前」「昨天」。 */
export function relativeTime(value: string | null | undefined): string {
  const d = parseTime(value)
  if (!d) return ''
  const diff = Date.now() - d.getTime()
  if (diff < 60_000) return '刚刚'
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)} 分钟前`
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)} 小时前`
  const days = Math.floor(diff / 86_400_000)
  if (days === 1) return '昨天'
  if (days < 7) return `${days} 天前`
  return d.toLocaleDateString('zh-CN', { month: 'numeric', day: 'numeric' })
}

/** 文件大小可读化。 */
export function formatBytes(bytes: number): string {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  return `${(bytes / 1024 ** i).toFixed(i === 0 ? 0 : 1)} ${units[i]}`
}
