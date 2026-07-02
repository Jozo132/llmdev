/**
 * Minimal dependency-free syntax highlighter for chat code blocks.
 * Escapes HTML first (XSS-safe), then wraps tokens in Tailwind-colored spans.
 */
const KEYWORDS =
  /\b(const|let|var|function|return|if|else|for|while|class|new|import|export|from|async|await|try|catch|throw|typeof|instanceof|def|self|lambda|None|True|False|print|struct|void|int|float|template)\b/g;

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function highlight(code: string): string {
  let out = escapeHtml(code);
  // Order matters: comments/strings first so keywords inside them stay plain.
  out = out.replace(
    /(\/\/[^\n]*|#[^\n]*|\/\*[\s\S]*?\*\/)/g,
    '<span class="text-slate-500 italic">$1</span>'
  );
  out = out.replace(
    /(&quot;.*?&quot;|'[^'\n]*'|`[^`]*`)/g,
    '<span class="text-emerald-300">$1</span>'
  );
  out = out.replace(KEYWORDS, '<span class="text-violet-300">$1</span>');
  out = out.replace(/\b(\d+\.?\d*)\b/g, '<span class="text-amber-300">$1</span>');
  return out;
}

export interface MessageSegment {
  kind: "text" | "code";
  content: string;
  lang?: string;
}

/** Split message text into prose and fenced code segments. */
export function segmentMessage(text: string): MessageSegment[] {
  const segments: MessageSegment[] = [];
  const re = /```(\w*)\n?([\s\S]*?)(?:```|$)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    if (m.index > last) segments.push({ kind: "text", content: text.slice(last, m.index) });
    segments.push({ kind: "code", content: m[2], lang: m[1] || "text" });
    last = re.lastIndex;
  }
  if (last < text.length) segments.push({ kind: "text", content: text.slice(last) });
  return segments;
}
