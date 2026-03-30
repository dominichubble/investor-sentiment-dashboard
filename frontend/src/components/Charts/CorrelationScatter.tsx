import React, { useMemo } from 'react';
import {
  ComposedChart,
  Scatter,
  Line,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Label,
  Cell,
} from 'recharts';
import type { TimeSeriesPoint } from '../../types';
import { formatDecimalDisplay, formatIntegerDisplay } from '../../utils/formatDisplay';
import { chartTheme } from './chartTheme';

interface CorrelationScatterProps {
  data: TimeSeriesPoint[];
  height?: number;
  correlationCoefficient?: number;
  /** Trailing window for net sentiment (for axis label); x uses trailing_net_sentiment from API. */
  trailingWindowDays?: number;
}

function linearRegression(xs: number[], ys: number[]): { a: number; b: number } {
  const n = xs.length;
  if (n < 2) return { a: 0, b: 0 };
  const mx = xs.reduce((s, x) => s + x, 0) / n;
  const my = ys.reduce((s, y) => s + y, 0) / n;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (xs[i] - mx) * (ys[i] - my);
    den += (xs[i] - mx) ** 2;
  }
  const b = den < 1e-14 ? 0 : num / den;
  const a = my - b * mx;
  return { a, b };
}

function quadrantFill(sentiment: number, retPct: number): string {
  if (sentiment >= 0 && retPct >= 0) return 'rgba(47, 158, 68, 0.82)';
  if (sentiment < 0 && retPct < 0) return 'rgba(224, 49, 49, 0.82)';
  if (sentiment >= 0 && retPct < 0) return 'rgba(245, 159, 0, 0.75)';
  return 'rgba(66, 99, 235, 0.75)';
}

const CorrelationScatter: React.FC<CorrelationScatterProps> = ({
  data,
  height = 360,
  correlationCoefficient,
  trailingWindowDays = 1,
}) => {
  const w = Math.max(1, Math.min(30, Math.floor(trailingWindowDays || 1)));

  const { scatterData, lineData, xDomain, yDomain, slope, intercept } = useMemo(() => {
    const raw = (data ?? [])
      .filter((p) => p.returns != null)
      .map((p) => ({
        sentiment:
          typeof p.trailing_net_sentiment === 'number'
            ? p.trailing_net_sentiment
            : p.net_sentiment,
        returns: (p.returns ?? 0) * 100,
        date: p.date,
        mentions: Math.max(0, p.mention_count ?? 0),
      }));
    if (!raw.length) {
      return {
        scatterData: [],
        lineData: [] as { sentiment: number; returns: number }[],
        xDomain: [-1, 1] as [number, number],
        yDomain: [-5, 5] as [number, number],
        slope: 0,
        intercept: 0,
      };
    }
    const xs = raw.map((d) => d.sentiment);
    const ys = raw.map((d) => d.returns);
    const { a, b } = linearRegression(xs, ys);
    const xPad = 0.08;
    const xMin = Math.max(-1, Math.min(...xs) - xPad);
    const xMax = Math.min(1, Math.max(...xs) + xPad);
    const yAbs = Math.max(3, ...ys.map((y) => Math.abs(y))) * 1.12;
    const yDom: [number, number] = [-yAbs, yAbs];
    const lineData = [
      { sentiment: xMin, returns: a + b * xMin },
      { sentiment: xMax, returns: a + b * xMax },
    ];
    return {
      scatterData: raw,
      lineData,
      xDomain: [xMin, xMax] as [number, number],
      yDomain: yDom,
      slope: b,
      intercept: a,
    };
  }, [data]);

  if (!scatterData.length) {
    return (
      <div
        style={{
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#868e96',
          fontSize: 14,
        }}
      >
        No overlapping days with returns — widen the date range if possible.
      </div>
    );
  }

  const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: any[] }) => {
    if (!active || !payload?.length) return null;
    const d = payload[0]?.payload;
    return (
      <div
        style={{
          background: chartTheme.tooltipBg,
          border: `1px solid ${chartTheme.tooltipBorder}`,
          borderRadius: 10,
          padding: '10px 14px',
          boxShadow: '0 6px 18px rgba(0,0,0,0.1)',
          fontSize: 13,
        }}
      >
        <p style={{ fontWeight: 600, margin: '0 0 6px', color: '#212529' }}>{d?.date}</p>
        <p style={{ margin: 0, color: '#495057' }}>
          {w > 1 ? `${w}-day trailing net` : 'Net sentiment'}:{' '}
          <strong>{formatDecimalDisplay(d?.sentiment, 3)}</strong>
        </p>
        <p style={{ margin: 0, color: '#495057' }}>
          Daily return: <strong>{formatDecimalDisplay(d?.returns, 2)}%</strong>
        </p>
        <p style={{ margin: '4px 0 0', color: '#868e96', fontSize: 12 }}>
          Mentions: {formatIntegerDisplay(d?.mentions)}
        </p>
      </div>
    );
  };

  const rColor =
    correlationCoefficient === undefined
      ? chartTheme.price
      : correlationCoefficient > 0.25
        ? chartTheme.sentimentPos
        : correlationCoefficient < -0.25
          ? chartTheme.sentimentNeg
          : chartTheme.axis;

  return (
    <div className="correlation-scatter-wrap">
      <ResponsiveContainer width="100%" height={height} minWidth={0}>
        <ComposedChart data={scatterData} margin={{ top: 12, right: 24, bottom: 28, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} />
          <XAxis
            type="number"
            dataKey="sentiment"
            domain={xDomain}
            tick={{ fontSize: 11, fill: chartTheme.axis }}
            tickLine={false}
            axisLine={{ stroke: chartTheme.grid }}
            tickFormatter={(v: number) => formatDecimalDisplay(v, 2)}
          >
            <Label
              value={
                w > 1
                  ? `Trailing net sentiment (${w}-day mean)`
                  : 'Net sentiment (that day)'
              }
              position="bottom"
              offset={12}
              style={{ fontSize: 12, fill: '#495057', fontWeight: 500 }}
            />
          </XAxis>
          <YAxis
            type="number"
            dataKey="returns"
            domain={yDomain}
            tick={{ fontSize: 11, fill: chartTheme.axis }}
            tickLine={false}
            axisLine={{ stroke: chartTheme.grid }}
            tickFormatter={(v: number) => `${Math.round(Number(v))}%`}
          >
            <Label
              value="Daily return (%)"
              angle={-90}
              position="insideLeft"
              style={{ fontSize: 12, fill: '#495057', fontWeight: 500 }}
            />
          </YAxis>
          <ZAxis type="number" dataKey="mentions" range={[48, 320]} name="Mentions" />
          <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '4 4' }} />
          <ReferenceLine x={0} stroke={chartTheme.grid} strokeDasharray="4 4" />
          <ReferenceLine y={0} stroke={chartTheme.grid} strokeDasharray="4 4" />

          <Scatter name="Days" isAnimationActive={false}>
            {scatterData.map((entry, index) => (
              <Cell key={`c-${index}`} fill={quadrantFill(entry.sentiment, entry.returns)} />
            ))}
          </Scatter>

          {lineData.length === 2 && scatterData.length >= 2 && (
            <Line
              data={lineData}
              dataKey="returns"
              stroke="#495057"
              strokeWidth={2}
              strokeDasharray="7 5"
              dot={false}
              name="OLS fit"
              isAnimationActive={false}
              legendType="line"
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>

      <div className="correlation-scatter-legend">
        <span className="correlation-scatter-legend__item">
          <i className="correlation-scatter-legend__swatch" style={{ background: 'rgba(47, 158, 68, 0.82)' }} />
          +Sentiment / +Return
        </span>
        <span className="correlation-scatter-legend__item">
          <i className="correlation-scatter-legend__swatch" style={{ background: 'rgba(224, 49, 49, 0.82)' }} />
          −Sentiment / −Return
        </span>
        <span className="correlation-scatter-legend__item">
          <i className="correlation-scatter-legend__swatch" style={{ background: 'rgba(245, 159, 0, 0.75)' }} />
          Mixed
        </span>
        <span className="correlation-scatter-legend__item correlation-scatter-legend__item--note">
          Point size ∝ mentions
        </span>
      </div>

      <div className="correlation-scatter-footer">
        {correlationCoefficient !== undefined && (
          <span>
            Pearson <em>r</em> ={' '}
            <strong style={{ color: rColor }}>{formatDecimalDisplay(correlationCoefficient, 4)}</strong>
          </span>
        )}
        {scatterData.length >= 2 && (
          <span className="correlation-scatter-footer__fit">
            OLS slope: <strong>{formatDecimalDisplay(slope, 3)}</strong>% per 1.0 sentiment
            {intercept !== 0 && (
              <>
                {' '}
                · intercept <strong>{formatDecimalDisplay(intercept, 2)}</strong>%
              </>
            )}
          </span>
        )}
      </div>
    </div>
  );
};

export default CorrelationScatter;
