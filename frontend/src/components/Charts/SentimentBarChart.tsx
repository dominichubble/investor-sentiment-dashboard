import React from 'react';
import { BarChart, Bar, ResponsiveContainer, Cell, XAxis } from 'recharts';

interface SentimentBarChartProps {
  positive: number;
  neutral: number;
  negative: number;
  height?: number;
}

const SentimentBarChart: React.FC<SentimentBarChartProps> = ({
  positive,
  neutral,
  negative,
  height = 80
}) => {
  const data = [
    { name: 'Positive', value: positive, color: '#7aac86' },
    { name: 'Neutral', value: neutral, color: '#8e94a0' },
    { name: 'Negative', value: negative, color: '#cb6e68' }
  ];

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data}>
        <XAxis 
          dataKey="name" 
          tick={{ fontSize: 10, fill: '#666' }}
          tickLine={false}
          axisLine={false}
        />
        <Bar dataKey="value" radius={[4, 4, 0, 0]}>
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
};

export default SentimentBarChart;
