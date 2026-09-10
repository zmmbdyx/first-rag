/**
 * Markdown 渲染：标题 / 列表 / 表格 / 引用 / 链接 / 代码块。
 *
 * - remark-gfm 提供表格、删除线、任务列表等 GFM 扩展；
 * - 代码块用 react-syntax-highlighter 高亮，右上角带语言标签与一键复制；
 * - 未知语言静默降级为纯文本，不让高亮失败把整条回答搞崩。
 */

import { memo } from 'react'
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { useCopy } from '@/hooks/useCopy'
import { useAppStore } from '@/store/useAppStore'
import { IconCheck, IconCopy } from '@/components/Icons'

/** 语言名 -> 展示用标签。 */
const LANG_LABEL: Record<string, string> = {
  ts: 'TypeScript',
  tsx: 'TSX',
  js: 'JavaScript',
  jsx: 'JSX',
  py: 'Python',
  python: 'Python',
  sh: 'Bash',
  bash: 'Bash',
  shell: 'Shell',
  json: 'JSON',
  yaml: 'YAML',
  yml: 'YAML',
  sql: 'SQL',
  go: 'Go',
  rust: 'Rust',
  java: 'Java',
  cpp: 'C++',
  c: 'C',
  css: 'CSS',
  html: 'HTML',
  md: 'Markdown',
  text: 'Text',
}

/** 代码块：头部（语言 + 复制）+ 高亮正文。 */
function CodeBlock({ language, code }: { language: string; code: string }) {
  const theme = useAppStore((s) => s.theme)
  const { copied, copy } = useCopy()
  const label = LANG_LABEL[language] ?? language ?? 'Text'

  return (
    <div className="code-block">
      <div className="flex items-center justify-between border-b border-line bg-surface px-3.5 py-2">
        <span className="font-mono text-[12px] font-medium text-ink-muted">{label}</span>
        <button
          type="button"
          onClick={() => copy(code)}
          className="icon-btn h-7 gap-1.5 px-2 text-[12px]"
          aria-label={copied ? '已复制' : '复制代码'}
          title={copied ? '已复制' : '复制代码'}
        >
          {copied ? <IconCheck size={14} /> : <IconCopy size={14} />}
          <span>{copied ? '已复制' : '复制'}</span>
        </button>
      </div>
      <SyntaxHighlighter
        language={language || 'text'}
        style={theme === 'dark' ? oneDark : oneLight}
        customStyle={{
          margin: 0,
          padding: '1rem',
          background: 'transparent',
          fontSize: '13px',
          lineHeight: '1.6',
        }}
        codeTagProps={{ style: { fontFamily: 'var(--font-mono, ui-monospace, monospace)' } }}
        wrapLongLines={false}
        PreTag="div"
      >
        {code}
      </SyntaxHighlighter>
    </div>
  )
}

/** 从 className（如 "language-python"）里取出语言名。 */
function langOf(className?: string): string {
  const m = /language-([\w+-]+)/.exec(className ?? '')
  return m ? m[1].toLowerCase() : ''
}

const components: Components = {
  // 代码：区分行内与块级。块级由自定义的 pre 负责渲染，这里只透传。
  code({ className, children, ...rest }) {
    const language = langOf(className)
    const text = String(children ?? '').replace(/\n$/, '')
    // 有 language-xxx 或内容含换行 => 块级代码
    const isBlock = Boolean(language) || text.includes('\n')
    if (isBlock) {
      return <CodeBlock language={language} code={text} />
    }
    return (
      <code className={className} {...rest}>
        {children}
      </code>
    )
  },
  // 块级代码外面包了一层 <pre>，去掉它以免和 CodeBlock 的样式打架
  pre({ children }) {
    return <>{children}</>
  },
  // 宽表格包一层可横向滚动的容器
  table({ children }) {
    return (
      <div className="table-wrap">
        <table>{children}</table>
      </div>
    )
  },
  a({ href, children }) {
    return (
      <a href={href} target="_blank" rel="noreferrer noopener">
        {children}
      </a>
    )
  },
}

export const MarkdownRenderer = memo(function MarkdownRenderer({
  content,
  className = '',
}: {
  content: string
  className?: string
}) {
  return (
    <div className={`md-body ${className}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {content}
      </ReactMarkdown>
    </div>
  )
})
