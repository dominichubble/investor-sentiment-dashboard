import React from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  Bar,
  Line,
} from 'recharts';
import type { EmotionTimelinePoint } from '../../types';
import { formatIntegerDisplay } from '../../utils/formatDisplay';
import { chartTheme } from './chartTheme';

const EMOTION_COLORS: Record<string, string> = {
  fear: '#d9485f',
  optimism: '#2f9e44',
  uncertainty: '#f08c00',
  confidence: '#1971c2',
  skepticism: '#7048e8',
  mixed: '#868e96',
};

interface Props {
  data: EmotionTimelinePoint[];
  height?: number;
}

const ORDER = ['fear', 'optimism', 'uncertainty', 'confidence', 'skepticism', 'mixed'];

const EmotionTimelineChart: React.FC<Props> = ({ data, height = 320 }) => {
  const chartData = data.map((point) => ({
    date: point.date,
    total_mentions: point.total_mentions,
    ...point.counts,
  }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={chartData} margin={{ top: 8, right: 24, left: 8, bottom: 12 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} />
        <XAxis dataKey="date" tick={{ fontSize: 11, fill: chartTheme.axis }} />
        <YAxis
          yAxisId="left"
          tick={{ fontSize: 11, fill: chartTheme.axis }}
          allowDecimals={false}
          tickFormatter={(v: number) => formatIntegerDisplay(v)}
        />
        <YAxis
          yAxisId="right"
          orientation="right"
          tick={{ fontSize: 11, fill: chartTheme.axis }}
          allowDecimals={false}
          tickFormatter={(v: number) => formatIntegerDisplay(v)}
        />
        <Tooltip
          contentStyle={{
            borderRadius: 8,
            border: `1px solid ${chartTheme.tooltipBorder}`,
            fontSize: 13,
          }}
        />
        <Legend />
        {ORDER.map((emotion) => (
          <Bar
            key={emotion}
            yAxisId="left"
            dataKey={emotion}
            stackId="emotion"
            fill={EMOTION_COLORS[emotion]}
            radius={emotion === 'mixed' ? [4, 4, 0, 0] : 0}
            name={emotion}
          />
        ))}
        <Line
          yAxisId="right"
          type="monotone"
          dataKey="total_mentions"
          stroke={chartTheme.rollingLine}
          strokeWidth={2}
          dot={false}
          name="total mentions"
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
};

export default EmotionTimelineChart;
