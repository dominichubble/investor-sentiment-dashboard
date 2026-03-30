import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
  LabelList,
} from 'recharts';
import type { LagResult } from '../../types';
import { formatDecimalDisplay } from '../../utils/formatDisplay';
import { chartTheme } from './chartTheme';

interface LagChartProps {
  data: LagResult[];
  bestLag?: LagResult | null;
  height?: number;
}

const LagChart: React.FC<LagChartProps> = ({
  data,
  bestLag,
  height = 320,
}) => {
  if (!data || data.length === 0) {
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
        No lag analysis data available
      </div>
    );
  }

  const chartData = data.map((lag) => ({
    lag: lag.lag_days,
    label: lag.lag_days === 0 ? 'Same day' : lag.lag_days > 0 ? `+${lag.lag_days}d` : `${lag.lag_days}d`,
    correlation: lag.pearson_r ?? 0,
    significant: lag.significant ?? false,
    description: lag.description,
    pValue: lag.p_value,
    isBest: bestLag ? lag.lag_days === bestLag.lag_days : false,
  }));

  const getBarColor = (entry: (typeof chartData)[0]) => {
    if (entry.isBest) return chartTheme.price;
    if (!entry.significant) return '#dee2e6';
    return entry.correlation > 0 ? chartTheme.sentimentPos : chartTheme.sentimentNeg;
  };

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
        <p style={{ fontWeight: 600, margin: '0 0 6px', color: '#212529' }}>{d?.description}</p>
        <p style={{ margin: 0, color: '#495057' }}>
          Pearson <em>r</em>: <strong>{formatDecimalDisplay(d?.correlation, 4)}</strong>
        </p>
        <p style={{ margin: 0, color: '#868e96', fontSize: 12 }}>
          <em>p</em> = {d?.pValue != null ? Number(d.pValue).toExponential(2) : '—'}
        </p>
        <p
          style={{
            margin: '6px 0 0',
            color: d?.significant ? chartTheme.sentimentPos : chartTheme.sentimentNeg,
            fontWeight: 500,
          }}
        >
          {d?.significant ? 'Significant (α = 0.05)' : 'Not significant'}
        </p>
      </div>
    );
  };

  return (
    <div className="lag-chart-wrap">
      <ResponsiveContainer width="100%" height={height} minWidth={0}>
        <BarChart data={chartData} margin={{ top: 28, right: 12, left: 4, bottom: 8 }} barCategoryGap="18%">
          <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} vertical={false} />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 11, fill: chartTheme.axis }}
            tickLine={false}
            axisLine={{ stroke: chartTheme.grid }}
          />
          <YAxis
            domain={[-1, 1]}
            tick={{ fontSize: 11, fill: chartTheme.axis }}
            tickLine={false}
            axisLine={{ stroke: chartTheme.grid }}
            tickFormatter={(v) => formatDecimalDisplay(v, 1)}
            width={36}
            label={{
              value: 'Correlation (r)',
              angle: -90,
              position: 'insideLeft',
              style: { fontSize: 11, fill: chartTheme.axis },
            }}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(0,0,0,0.04)' }} />
          <ReferenceLine y={0} stroke={chartTheme.axis} strokeDasharray="4 4" strokeOpacity={0.75} />

          <Bar dataKey="correlation" radius={[6, 6, 0, 0]} maxBarSize={48}>
            <LabelList
              dataKey="correlation"
              position="top"
              formatter={(v: number | string) => {
                const n = Number(v);
                return Math.abs(n) < 0.01 ? '' : formatDecimalDisplay(n, 2);
              }}
              style={{ fontSize: 10, fill: '#495057', fontWeight: 600 }}
            />
            {chartData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={getBarColor(entry)}
                stroke={entry.isBest ? '#364fc7' : 'transparent'}
                strokeWidth={entry.isBest ? 2 : 0}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {bestLag && bestLag.pearson_r !== null && (
        <div className="lag-chart-caption">
          Strongest lag:{' '}
          <strong style={{ color: chartTheme.price }}>
            {bestLag.lag_days === 0
              ? 'same day'
              : bestLag.lag_days > 0
                ? `sentiment leads by ${bestLag.lag_days}d`
                : `price leads by ${Math.abs(bestLag.lag_days)}d`}
          </strong>{' '}
          (<em>r</em> = {formatDecimalDisplay(bestLag.pearson_r, 4)})
        </div>
      )}
    </div>
  );
};

export default LagChart;
