import { useId } from 'react';
import './BrandMark.css';

export interface BrandMarkProps {
  /** Pixel width/height (square). Omit to fill the parent (parent should set width/height). */
  size?: number;
  className?: string;
  /** When true, exposes an accessible name for standalone use. */
  labelled?: boolean;
}

/**
 * Sentiment Lab logomark: gradient tile with abstract “volume bars + sentiment arc”.
 */
export function BrandMark({ size, className, labelled = false }: BrandMarkProps) {
  const gradId = `brand-grad-${useId().replace(/:/g, '')}`;
  const dim = size ?? ('100%' as const);

  return (
    <svg
      className={['brand-mark', className].filter(Boolean).join(' ')}
      width={dim}
      height={dim}
      viewBox="0 0 40 40"
      role={labelled ? 'img' : 'presentation'}
      aria-hidden={labelled ? undefined : true}
      aria-label={labelled ? 'Sentiment Lab' : undefined}
      focusable="false"
    >
      <defs>
        <linearGradient id={gradId} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#4f46e5" />
          <stop offset="52%" stopColor="#6366f1" />
          <stop offset="100%" stopColor="#7c3aed" />
        </linearGradient>
      </defs>
      <rect width="40" height="40" rx="10" fill={`url(#${gradId})`} />
      <path
        d="M 6 27 Q 14 10 34 16"
        fill="none"
        stroke="rgba(255,255,255,0.28)"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path
        d="M 6 28 Q 15 12 34 19"
        fill="none"
        stroke="rgba(255,255,255,0.95)"
        strokeWidth="2.25"
        strokeLinecap="round"
      />
      <rect x="7.5" y="23" width="4.5" height="10" rx="1.25" fill="rgba(255,255,255,0.88)" />
      <rect x="14.25" y="17" width="4.5" height="16" rx="1.25" fill="rgba(255,255,255,0.95)" />
      <rect x="21" y="13" width="4.5" height="20" rx="1.25" fill="rgba(255,255,255,0.9)" />
      <rect x="27.75" y="9" width="4.5" height="24" rx="1.25" fill="rgba(255,255,255,0.82)" />
    </svg>
  );
}
