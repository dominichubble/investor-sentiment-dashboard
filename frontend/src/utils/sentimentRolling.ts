/**
 * Trailing (causal) simple moving average: each index uses only that day and prior days
 * within the window — no lookahead. Day i = mean(values[i - window + 1 .. i]).
 */
export function trailingRollingMean(values: number[], window: number): number[] {
  if (window <= 1) {
    return values.map((v) => v);
  }
  const out: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const start = Math.max(0, i - window + 1);
    const slice = values.slice(start, i + 1);
    const sum = slice.reduce((a, b) => a + b, 0);
    out.push(sum / slice.length);
  }
  return out;
}
