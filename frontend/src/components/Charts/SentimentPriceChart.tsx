import React from 'react';
import {
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
} from 'recharts';
import type { TimeSeriesPoint } from '../../types';

interface SentimentPriceChartProps {
  data: TimeSeriesPoint[];
  height?: number;
  showVolume?: boolean;
}

const SentimentPriceChart: React.FC<SentimentPriceChartProps> = ({
  data,
  height = 400,
  showVolume = true,
}) => {
  if (!data || data.length === 0) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#999' }}>
        No time-series data available
      </div>
    );
  }

  const chartData = data.map(point => ({
    ...point,
    date: point.date.slice(5), // MM-DD format
    fullDate: point.date,
  }));

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || payload.length === 0) return null;
    const d = payload[0]?.payload;
    return (
      <div style={{
        background: 'white',
        border: '1px solid #e0e0e0',
        borderRadius: 8,
        padding: '12px 16px',
        boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
        fontSize: 13,
        lineHeight: 1.6,
      }}>
        <p style={{ fontWeight: 600, margin: 0, marginBottom: 6 }}>{d?.fullDate}</p>
        <p style={{ margin: 0, color: '#5c7cfa' }}>
          Price: <strong>${d?.close?.toFixed(2)}</strong>
        </p>
        <p style={{ margin: 0, color: d?.net_sentiment >= 0 ? '#7aac86' : '#cb6e68' }}>
          Sentiment: <strong>{d?.net_sentiment?.toFixed(3)}</strong>
        </p>
        <p style={{ margin: 0, color: '#8e94a0' }}>
          Mentions: <strong>{d?.mention_count}</strong>
        </p>
        {d?.returns != null && (
          <p style={{ margin: 0, color: d.returns >= 0 ? '#7aac86' : '#cb6e68' }}>
            Return: <strong>{(d.returns * 100).toFixed(2)}%</strong>
          </p>
        )}
      </div>
    );
  };

  return (
    <ResponsiveContainer width="100%" height={height} minWidth={0}>
      <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 11, fill: '#888' }}
          tickLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          yAxisId="price"
          orientation="left"
          tick={{ fontSize: 11, fill: '#5c7cfa' }}
          tickFormatter={(v) => `$${v.toFixed(0)}`}
          label={{ value: 'Price ($)', angle: -90, position: 'insideLeft', fill: '#5c7cfa', fontSize: 12 }}
        />
        <YAxis
          yAxisId="sentiment"
          orientation="right"
          domain={[-1, 1]}
          tick={{ fontSize: 11, fill: '#7aac86' }}
          tickFormatter={(v) => v.toFixed(1)}
          label={{ value: 'Net Sentiment', angle: 90, position: 'insideRight', fill: '#7aac86', fontSize: 12 }}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ fontSize: 12 }} />

        {showVolume && (
          <Bar
            yAxisId="sentiment"
            dataKey="mention_count"
            fill="#e8ecf4"
            name="Mentions"
            barSize={8}
            radius={[2, 2, 0, 0]}
          />
        )}

        <Line
          yAxisId="price"
          type="monotone"
          dataKey="close"
          stroke="#5c7cfa"
          strokeWidth={2.5}
          dot={false}
          name="Stock Price"
          activeDot={{ r: 4 }}
        />

        <Area
          yAxisId="sentiment"
          type="monotone"
          dataKey="net_sentiment"
          stroke="#7aac86"
          fill="#7aac86"
          fillOpacity={0.15}
          strokeWidth={2}
          name="Net Sentiment"
          activeDot={{ r: 4 }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
};

export default SentimentPriceChart;
