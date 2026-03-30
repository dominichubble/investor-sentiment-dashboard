import React, { useMemo } from 'react';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  ReferenceLine,
} from 'recharts';
import { formatDecimalDisplay } from '../../utils/formatDisplay';
import { chartTheme } from './chartTheme';

export interface RollingCorrelationPoint {
  date: string;
  correlation: number;
}

interface RollingCorrelationChartProps {
  data: RollingCorrelationPoint[];
  height?: number;
  windowDays?: number;
}

const RollingCorrelationChart: React.FC<RollingCorrelationChartProps> = ({
  data,
  height = 320,
  windowDays = 14,
}) => {
  const chartData = useMemo(
    () =>
      data.map((d) => ({
        ...d,
        shortDate: d.date.slice(5),
        pos: d.correlation >= 0 ? d.correlation : 0,
        neg: d.correlation < 0 ? d.correlation : 0,
      })),
    [data],
  );

  if (!chartData.length) {
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
        No rolling correlation data
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={height} minWidth={0}>
      <ComposedChart
        data={chartData}
        margin={{ top: 12, right: 16, left: 4, bottom: 8 }}
      >
        <defs>
          <linearGradient id="rollingCorrPos" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={chartTheme.rollingPos} stopOpacity={0.35} />
            <stop offset="100%" stopColor={chartTheme.rollingPos} stopOpacity={0.02} />
          </linearGradient>
          <linearGradient id="rollingCorrNeg" x1="0" y1="1" x2="0" y2="0">
            <stop offset="0%" stopColor={chartTheme.rollingNeg} stopOpacity={0.35} />
            <stop offset="100%" stopColor={chartTheme.rollingNeg} stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} vertical={false} />
        <XAxis
          dataKey="shortDate"
          tick={{ fontSize: 11, fill: chartTheme.axis }}
          tickLine={false}
          axisLine={{ stroke: chartTheme.grid }}
          interval="preserveStartEnd"
          minTickGap={32}
        />
        <YAxis
          domain={[-1, 1]}
          tick={{ fontSize: 11, fill: chartTheme.axis }}
          tickLine={false}
          axisLine={{ stroke: chartTheme.grid }}
          tickFormatter={(v: number) => formatDecimalDisplay(v, 1)}
          width={44}
          label={{
            value: `Rolling r (${windowDays}d)`,
            angle: -90,
            position: 'insideLeft',
            style: { fontSize: 11, fill: chartTheme.axis },
          }}
        />
        <Tooltip
          contentStyle={{
            background: chartTheme.tooltipBg,
            border: `1px solid ${chartTheme.tooltipBorder}`,
            borderRadius: 10,
            fontSize: 12,
            boxShadow: '0 4px 14px rgba(0,0,0,0.08)',
          }}
          labelFormatter={(_label, payload) => {
            const row = payload?.[0]?.payload as RollingCorrelationPoint | undefined;
            return row?.date ?? '';
          }}
          formatter={(value: number) => [formatDecimalDisplay(value, 4), 'Correlation']}
        />
        <ReferenceLine y={0} stroke={chartTheme.axis} strokeDasharray="4 4" strokeOpacity={0.7} />
        <ReferenceLine
          y={0.2}
          stroke={chartTheme.axis}
          strokeDasharray="2 6"
          strokeOpacity={0.35}
        />
        <ReferenceLine
          y={-0.2}
          stroke={chartTheme.axis}
          strokeDasharray="2 6"
          strokeOpacity={0.35}
        />
        <Area
          type="monotone"
          dataKey="pos"
          stroke="none"
          fill="url(#rollingCorrPos)"
          isAnimationActive={false}
        />
        <Area
          type="monotone"
          dataKey="neg"
          stroke="none"
          fill="url(#rollingCorrNeg)"
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="correlation"
          stroke={chartTheme.rollingLine}
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 5, fill: chartTheme.rollingLine, stroke: '#fff', strokeWidth: 2 }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
};

export default RollingCorrelationChart;
