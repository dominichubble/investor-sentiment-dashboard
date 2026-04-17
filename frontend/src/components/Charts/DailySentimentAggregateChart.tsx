import React, { useMemo } from 'react';
import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { TimeSeriesPoint } from '../../types';
import { formatDecimalDisplay, formatIntegerDisplay } from '../../utils/formatDisplay';
import { chartTheme } from './chartTheme';

const NEUTRAL_BAND = '#94a3b8';

interface DailySentimentAggregateChartProps {
  data: TimeSeriesPoint[];
  height?: number;
}

const DailySentimentAggregateChart: React.FC<DailySentimentAggregateChartProps> = ({
  data,
  height = 380,
}) => {
  const chartData = useMemo(
    () =>
      (data ?? []).map((p) => ({
        ...p,
        shortDate: p.date.slice(5),
        fullDate: p.date,
      })),
    [data],
  );

  const tickEvery = useMemo(() => {
    const n = chartData.length;
    if (n <= 15) return 0;
    if (n <= 40) return Math.ceil(n / 14) - 1;
    return Math.ceil(n / 12) - 1;
  }, [chartData.length]);

  if (!chartData.length) {
    return (
      <div
        style={{
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: chartTheme.axis,
          fontSize: 14,
        }}
      >
        No daily sentiment data for this range.
      </div>
    );
  }

  const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: any[] }) => {
    if (!active || !payload?.length) return null;
    const d = payload[0]?.payload as TimeSeriesPoint & { fullDate: string };
    return (
      <div
        style={{
          background: chartTheme.tooltipBg,
          border: `1px solid ${chartTheme.tooltipBorder}`,
          borderRadius: 10,
          padding: '12px 16px',
          boxShadow: '0 6px 20px rgba(0,0,0,0.1)',
          fontSize: 13,
          lineHeight: 1.55,
        }}
      >
        <p style={{ fontWeight: 700, margin: '0 0 8px', color: '#0f172a' }}>{d.fullDate}</p>
        <p style={{ margin: 0, color: '#475569' }}>
          Net sentiment:{' '}
          <strong style={{ color: d.net_sentiment >= 0 ? chartTheme.sentimentPos : chartTheme.sentimentNeg }}>
            {formatDecimalDisplay(d.net_sentiment, 3)}
          </strong>
        </p>
        <p style={{ margin: 0, color: '#475569' }}>
          Avg score: <strong>{formatDecimalDisplay(d.avg_sentiment_score, 3)}</strong>
        </p>
        <p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 12 }}>
          Mentions: <strong>{formatIntegerDisplay(d.mention_count)}</strong>
        </p>
        <p style={{ margin: '8px 0 0', fontSize: 11, color: '#94a3b8', borderTop: '1px solid #e2e8f0', paddingTop: 8 }}>
          Mix: +{Math.round((d.positive_ratio ?? 0) * 100)}% / ○{Math.round((d.neutral_ratio ?? 0) * 100)}% / −
          {Math.round((d.negative_ratio ?? 0) * 100)}%
        </p>
      </div>
    );
  };

  return (
    <ResponsiveContainer width="100%" height={height} minWidth={0} debounce={120}>
      <ComposedChart data={chartData} margin={{ top: 16, right: 52, left: 4, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} vertical={false} />
        <XAxis
          dataKey="shortDate"
          tick={{ fontSize: 11, fill: chartTheme.axis }}
          tickLine={false}
          axisLine={{ stroke: chartTheme.grid }}
          interval={tickEvery > 0 ? tickEvery : 'preserveStartEnd'}
          minTickGap={28}
        />
        <YAxis
          yAxisId="share"
          domain={[0, 1]}
          tick={{ fontSize: 11, fill: chartTheme.axis }}
          tickLine={false}
          axisLine={{ stroke: chartTheme.grid }}
          tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
          width={44}
          label={{
            value: 'Daily mention mix',
            angle: -90,
            position: 'insideLeft',
            style: { fontSize: 11, fill: chartTheme.axis, fontWeight: 600 },
          }}
        />
        <YAxis
          yAxisId="net"
          orientation="right"
          domain={[-1, 1]}
          tick={{ fontSize: 11, fill: '#4f46e5' }}
          tickLine={false}
          axisLine={{ stroke: chartTheme.grid }}
          tickFormatter={(v: number) => formatDecimalDisplay(v, 1)}
          width={40}
          label={{
            value: 'Net sentiment',
            angle: 90,
            position: 'insideRight',
            style: { fontSize: 11, fill: '#4f46e5', fontWeight: 600 },
          }}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend
          wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
          formatter={(value) => <span style={{ color: '#475569' }}>{value}</span>}
        />

        <Area
          yAxisId="share"
          type="monotone"
          dataKey="positive_ratio"
          name="Positive share"
          stackId="mix"
          stroke={chartTheme.sentimentPos}
          strokeWidth={0.5}
          fill={chartTheme.sentimentPos}
          fillOpacity={0.85}
        />
        <Area
          yAxisId="share"
          type="monotone"
          dataKey="neutral_ratio"
          name="Neutral share"
          stackId="mix"
          stroke={NEUTRAL_BAND}
          strokeWidth={0.5}
          fill={NEUTRAL_BAND}
          fillOpacity={0.45}
        />
        <Area
          yAxisId="share"
          type="monotone"
          dataKey="negative_ratio"
          name="Negative share"
          stackId="mix"
          stroke={chartTheme.sentimentNeg}
          strokeWidth={0.5}
          fill={chartTheme.sentimentNeg}
          fillOpacity={0.85}
        />

        <Line
          yAxisId="net"
          type="monotone"
          dataKey="net_sentiment"
          name="Net sentiment"
          stroke="#4f46e5"
          strokeWidth={2.5}
          dot={false}
          activeDot={{ r: 5, stroke: '#fff', strokeWidth: 2 }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
};

export default DailySentimentAggregateChart;
