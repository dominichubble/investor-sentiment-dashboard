/**
 * Human-readable numbers for charts and UI (avoids float artifacts like 3593.1600000000003).
 */

export function formatIntegerDisplay(value: unknown): string {
  const n = Number(value);
  if (!Number.isFinite(n)) return '—';
  return Math.round(n).toLocaleString();
}

/**
 * Upper bound for count / mention axes: pad, ceil, then snap up to a clean step (…, 100, 200, 500, 1k, 2k, …)
 * so domains stay round and Recharts is less likely to pick awkward float ticks.
 */
export function paddedNiceCountAxisMax(baseMax: number, pad = 1.1): number {
  const raw = Math.max(1, baseMax) * pad;
  const ceilRaw = Math.ceil(raw);
  const log = Math.log10(ceilRaw);
  if (!Number.isFinite(log) || log < 0) return ceilRaw;
  const magnitude = 10 ** Math.floor(log);
  return Math.ceil(ceilRaw / magnitude) * magnitude;
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
