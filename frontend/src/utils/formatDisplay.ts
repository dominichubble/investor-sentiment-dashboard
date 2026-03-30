/**
 * Human-readable numbers for charts and UI (avoids float artifacts like 3593.1600000000003).
 */

export function formatIntegerDisplay(value: unknown): string {
  const n = Number(value);
  if (!Number.isFinite(n)) return '—';
  return Math.round(n).toLocaleString();
}

/** Rounds then formats with at most `decimals` fractional digits (stable for axis ticks). */
export function formatDecimalDisplay(value: unknown, decimals: number): string {
  const n = Number(value);
  if (!Number.isFinite(n)) return '—';
  if (decimals <= 0) return Math.round(n).toLocaleString();
  const f = 10 ** decimals;
  const r = Math.round(n * f) / f;
  return r.toLocaleString(undefined, {
    maximumFractionDigits: decimals,
    minimumFractionDigits: 0,
  });
}
