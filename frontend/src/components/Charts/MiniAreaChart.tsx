import React from 'react';
import { AreaChart, Area, ResponsiveContainer, YAxis } from 'recharts';

interface MiniAreaChartProps {
  data: number[];
  color: string;
  height?: number;
}

const MiniAreaChart: React.FC<MiniAreaChartProps> = ({ 
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
      <AreaChart data={chartData}>
        <YAxis hide domain={['auto', 'auto']} />
        <defs>
          <linearGradient id={`gradient-${color}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={color} stopOpacity={0.8}/>
            <stop offset="95%" stopColor={color} stopOpacity={0.1}/>
          </linearGradient>
        </defs>
        <Area 
          type="monotone" 
          dataKey="value" 
          stroke={color} 
          strokeWidth={2}
          fill={`url(#gradient-${color})`}
          isAnimationActive={true}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
};

export default MiniAreaChart;
