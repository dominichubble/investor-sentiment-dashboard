import React from 'react';
import { LineChart, Line, ResponsiveContainer, YAxis } from 'recharts';

interface MiniLineChartProps {
  data: number[];
  color: string;
  height?: number;
}

const MiniLineChart: React.FC<MiniLineChartProps> = ({ 
  data, 
  color,
  height = 60 
}) => {
  if (!data || data.length === 0) return null;

  const chartData = data.map((value, index) => ({
    value,
    index
  }));

  return (
    <ResponsiveContainer width="100%" height={height} minWidth={0} debounce={120}>
      <LineChart data={chartData}>
        <YAxis hide domain={['auto', 'auto']} />
        <Line 
          type="monotone" 
          dataKey="value" 
          stroke={color} 
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default MiniLineChart;
