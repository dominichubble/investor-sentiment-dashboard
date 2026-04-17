import React, { useMemo } from 'react';
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
} from 'recharts';
import type { DailyTrendPoint } from '../../services/api';
import { formatDecimalDisplay, formatIntegerDisplay, paddedNiceCountAxisMax } from '../../utils/formatDisplay';
import { chartTheme } from './chartTheme';

interface GlobalMarketSentimentChartProps {
  data: DailyTrendPoint[];
  height?: number;
}

const GlobalMarketSentimentChart: React.FC<GlobalMarketSentimentChartProps> = ({
  data,
  height = 300,
}) => {
  const { chartData, maxCount } = useMemo(() => {
    const rows = (data ?? []).map((p) => ({
      ...p,
      shortDate: p.date.slice(5),
      fullDate: p.date,
    }));
    const maxC = Math.max(1, ...rows.map((r) => r.count || 0));
    return { chartData: rows, maxCount: paddedNiceCountAxisMax(maxC, 1.08) };
  }, [data]);

  const tickEvery = useMemo(() => {
    const n = chartData.length;
    if (n <= 18) return 0;
    return Math.ceil(n / 14) - 1;
  }, [chartData.length]);

  if (!chartData.length) {
    return null;
  }

  const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: any[] }) => {
    if (!active || !payload?.length) return null;
    const d = payload[0]?.payload as DailyTrendPoint & { fullDate: string };
    return (
      <div
        style={{
          background: chartTheme.tooltipBg,
          border: `1px solid ${chartTheme.tooltipBorder}`,
          borderRadius: 10,
          padding: '12px 16px',
          fontSize: 13,
          boxShadow: '0 6px 20px rgba(0,0,0,0.1)',
        }}
      >
        <p style={{ fontWeight: 700, margin: '0 0 6px', color: '#0f172a' }}>{d.fullDate}</p>
        <p style={{ margin: 0, color: '#475569' }}>
          Net sentiment:{' '}
          <strong style={{ color: d.net_sentiment >= 0 ? chartTheme.sentimentPos : chartTheme.sentimentNeg }}>
            {formatDecimalDisplay(d.net_sentiment, 3)}
          </strong>
        </p>
        <p style={{ margin: 0, color: '#64748b', fontSize: 12 }}>
          Stock-level mentions: <strong>{formatIntegerDisplay(d.count)}</strong>
        </p>
      </div>
    );
  };

  return (
    <ResponsiveContainer width="100%" height={height} minWidth={0} debounce={120}>
      <ComposedChart data={chartData} margin={{ top: 12, right: 48, left: 4, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} vertical={false} />
        <XAxis
          dataKey="shortDate"
          tick={{ fontSize: 11, fill: chartTheme.axis }}
          tickLine={false}
          axisLine={{ stroke: chartTheme.grid }}
          interval={tickEvery > 0 ? tickEvery : 'preserveStartEnd'}
          minTickGap={24}
        />
        <YAxis
          yAxisId="count"
          hide
          domain={[0, maxCount]}
          allowDecimals={false}
        />
        <YAxis
          yAxisId="net"
          orientation="right"
          domain={[-1, 1]}
          tick={{ fontSize: 11, fill: '#4f46e5' }}
          tickLine={false}
          width={40}
          tickFormatter={(v: number) => formatDecimalDisplay(v, 1)}
          label={{
            value: 'Net (−1…1)',
            angle: 90,
            position: 'insideRight',
            style: { fontSize: 10, fill: '#4f46e5', fontWeight: 600 },
          }}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ fontSize: 12, paddingTop: 6 }} />
        <Bar
          yAxisId="count"
          dataKey="count"
          name="Mentions / day"
          fill={chartTheme.volumeBar}
          radius={[3, 3, 0, 0]}
          barSize={Math.min(12, Math.max(4, 420 / chartData.length))}
        />
        <Line
          yAxisId="net"
          type="monotone"
          dataKey="net_sentiment"
          name="Net sentiment"
          stroke="#4f46e5"
          strokeWidth={2.5}
          dot={false}
          activeDot={{ r: 4 }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
};

export default GlobalMarketSentimentChart;
