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
} from 'recharts';
import type { LagResult } from '../../types';

interface LagChartProps {
  data: LagResult[];
  bestLag?: LagResult | null;
  height?: number;
}

const LagChart: React.FC<LagChartProps> = ({
  data,
  bestLag,
  height = 300,
}) => {
  if (!data || data.length === 0) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#999' }}>
        No lag analysis data available
      </div>
    );
  }

  const chartData = data.map(lag => ({
    lag: lag.lag_days,
    label: lag.lag_days === 0 ? 'Same day' : lag.lag_days > 0 ? `+${lag.lag_days}d` : `${lag.lag_days}d`,
    correlation: lag.pearson_r ?? 0,
    significant: lag.significant ?? false,
    description: lag.description,
    pValue: lag.p_value,
    isBest: bestLag ? lag.lag_days === bestLag.lag_days : false,
  }));

  const getBarColor = (entry: typeof chartData[0]) => {
    if (entry.isBest) return '#5c7cfa';
    if (!entry.significant) return '#d0d4db';
    return entry.correlation > 0 ? '#7aac86' : '#cb6e68';
  };

  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload || payload.length === 0) return null;
    const d = payload[0]?.payload;
    return (
      <div style={{
        background: 'white',
        border: '1px solid #e0e0e0',
        borderRadius: 8,
        padding: '10px 14px',
        boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
        fontSize: 13,
      }}>
        <p style={{ fontWeight: 600, margin: 0, marginBottom: 4 }}>{d?.description}</p>
        <p style={{ margin: 0 }}>
          Correlation: <strong>{d?.correlation?.toFixed(4)}</strong>
        </p>
        <p style={{ margin: 0, color: '#999' }}>
          p-value: {d?.pValue?.toFixed(6)}
        </p>
        <p style={{ margin: 0, color: d?.significant ? '#7aac86' : '#cb6e68' }}>
          {d?.significant ? 'Statistically significant' : 'Not significant'}
        </p>
      </div>
    );
  };

  return (
    <div>
      <ResponsiveContainer width="100%" height={height} minWidth={0}>
        <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 11, fill: '#888' }}
            tickLine={false}
          />
          <YAxis
            domain={[-1, 1]}
            tick={{ fontSize: 11, fill: '#888' }}
            tickFormatter={(v) => v.toFixed(1)}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={0} stroke="#ccc" />

          <Bar dataKey="correlation" radius={[4, 4, 0, 0]}>
            {chartData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={getBarColor(entry)}
                stroke={entry.isBest ? '#3b5bdb' : 'none'}
                strokeWidth={entry.isBest ? 2 : 0}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {bestLag && bestLag.pearson_r !== null && (
        <div style={{
          textAlign: 'center',
          marginTop: 8,
          fontSize: 13,
          color: '#666',
        }}>
          Strongest correlation at <strong style={{ color: '#5c7cfa' }}>
            {bestLag.lag_days === 0
              ? 'same day'
              : bestLag.lag_days > 0
                ? `${bestLag.lag_days}-day lead`
                : `${Math.abs(bestLag.lag_days)}-day lag`}
          </strong> (r = {bestLag.pearson_r.toFixed(4)})
        </div>
      )}
    </div>
  );
};

export default LagChart;
