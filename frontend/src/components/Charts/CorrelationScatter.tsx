import React from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Label,
} from 'recharts';
import type { TimeSeriesPoint } from '../../types';

interface CorrelationScatterProps {
  data: TimeSeriesPoint[];
  height?: number;
  correlationCoefficient?: number;
}

const CorrelationScatter: React.FC<CorrelationScatterProps> = ({
  data,
  height = 350,
  correlationCoefficient,
}) => {
  if (!data || data.length === 0) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#999' }}>
        No data available for scatter plot
      </div>
    );
  }

  const scatterData = data
    .filter(p => p.returns != null)
    .map(p => ({
      sentiment: p.net_sentiment,
      returns: (p.returns ?? 0) * 100,
      date: p.date,
      mentions: p.mention_count,
    }));

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
        <p style={{ fontWeight: 600, margin: 0, marginBottom: 4 }}>{d?.date}</p>
        <p style={{ margin: 0 }}>Sentiment: <strong>{d?.sentiment?.toFixed(3)}</strong></p>
        <p style={{ margin: 0 }}>Return: <strong>{d?.returns?.toFixed(2)}%</strong></p>
        <p style={{ margin: 0, color: '#999' }}>Mentions: {d?.mentions}</p>
      </div>
    );
  };

  const getColor = (r?: number): string => {
    if (r === undefined) return '#5c7cfa';
    if (r > 0.3) return '#7aac86';
    if (r < -0.3) return '#cb6e68';
    return '#5c7cfa';
  };

  return (
    <div>
      <ResponsiveContainer width="100%" height={height} minWidth={0}>
        <ScatterChart margin={{ top: 10, right: 30, bottom: 20, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis
            type="number"
            dataKey="sentiment"
            name="Net Sentiment"
            tick={{ fontSize: 11, fill: '#888' }}
            domain={[-1, 1]}
          >
            <Label value="Net Sentiment" position="bottom" offset={0} style={{ fontSize: 12, fill: '#666' }} />
          </XAxis>
          <YAxis
            type="number"
            dataKey="returns"
            name="Daily Return (%)"
            tick={{ fontSize: 11, fill: '#888' }}
            tickFormatter={(v) => `${v.toFixed(1)}%`}
          >
            <Label value="Daily Return (%)" angle={-90} position="insideLeft" style={{ fontSize: 12, fill: '#666' }} />
          </YAxis>
          <Tooltip content={<CustomTooltip />} />

          <ReferenceLine x={0} stroke="#ddd" strokeDasharray="3 3" yAxisId={0} />
          <ReferenceLine y={0} stroke="#ddd" strokeDasharray="3 3" />

          <Scatter
            data={scatterData}
            fill={getColor(correlationCoefficient)}
            fillOpacity={0.7}
            r={5}
          />
        </ScatterChart>
      </ResponsiveContainer>

      {correlationCoefficient !== undefined && (
        <div style={{
          textAlign: 'center',
          marginTop: 8,
          fontSize: 13,
          color: '#666',
        }}>
          Pearson r = <strong style={{ color: getColor(correlationCoefficient) }}>
            {correlationCoefficient.toFixed(4)}
          </strong>
        </div>
      )}
    </div>
  );
};

export default CorrelationScatter;
