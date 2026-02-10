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
  const chartData = data.map((value, index) => ({
    value,
    index
  }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={chartData}>
        <YAxis hide domain={['auto', 'auto']} />
        <Line 
          type="monotone" 
          dataKey="value" 
          stroke={color} 
          strokeWidth={2}
          dot={false}
          isAnimationActive={true}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default MiniLineChart;
