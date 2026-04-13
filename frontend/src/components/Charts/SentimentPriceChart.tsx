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
  Area,
} from 'recharts';
import type { TimeSeriesPoint } from '../../types';
import { chartTheme } from './chartTheme';
import { formatDecimalDisplay, formatIntegerDisplay, paddedNiceCountAxisMax } from '../../utils/formatDisplay';
import { trailingRollingMean } from '../../utils/sentimentRolling';

interface SentimentPriceChartProps {
  data: TimeSeriesPoint[];
  height?: number;
  showVolume?: boolean;
  /** 1 = same-day net only; 3+ = trailing mean of net_sentiment over that many days (causal). */
  rollingWindowDays?: number;
  /** When equal to rolling window and points include trailing_net_sentiment, use API values (no client recompute). */
  apiTrailingDays?: number;
}

const SentimentPriceChart: React.FC<SentimentPriceChartProps> = ({
  data,
  height = 400,
  showVolume = true,
  rollingWindowDays = 3,
  apiTrailingDays,
}) => {
  const window = Math.max(1, Math.min(30, Math.floor(rollingWindowDays || 1)));

  const { chartData, maxMentions, tickEvery } = useMemo(() => {
    if (!data?.length) {
      return { chartData: [], maxMentions: 1, tickEvery: 1 };
    }
    const maxM = Math.max(1, ...data.map((p) => p.mention_count ?? 0));
    const n = data.length;
    const tickEvery = n > 45 ? Math.ceil(n / 12) : n > 20 ? 2 : 1;

    const useApi =
      apiTrailingDays === window &&
      data.every((p) => typeof p.trailing_net_sentiment === 'number');
    const nets = data.map((p) => p.net_sentiment);
    const rollingNet = useApi
      ? data.map((p) => p.trailing_net_sentiment)
      : trailingRollingMean(nets, window);

    const chartData = data.map((point, i) => ({
      ...point,
      shortDate: point.date.slice(5),
      fullDate: point.date,
      rolling_net_sentiment: rollingNet[i],
    }));
    return { chartData, maxMentions: paddedNiceCountAxisMax(maxM, 1.08), tickEvery };
  }, [data, window, apiTrailingDays]);

  if (!chartData.length) {
    return (
      <div
        style={{
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#868e96',
        }}
      >
        No time-series data available
      </div>
    );
  }

  const showRawOverlay = window > 1;

  const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: any[] }) => {
    if (!active || !payload?.length) return null;
    const d = payload[0]?.payload;
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
        <p style={{ fontWeight: 600, margin: '0 0 8px', color: '#212529' }}>{d?.fullDate}</p>
        <p style={{ margin: 0, color: chartTheme.price }}>
          Close: <strong>${formatDecimalDisplay(d?.close, 2)}</strong>
        </p>
        {showRawOverlay && (
          <p
            style={{
              margin: 0,
              color: d?.net_sentiment >= 0 ? chartTheme.sentimentPos : chartTheme.sentimentNeg,
              opacity: 0.85,
            }}
          >
            Same-day net: <strong>{formatDecimalDisplay(d?.net_sentiment, 3)}</strong>
          </p>
        )}
        <p
          style={{
            margin: 0,
            color:
              d?.rolling_net_sentiment >= 0 ? chartTheme.sentimentPos : chartTheme.sentimentNeg,
          }}
        >
          {window > 1 ? `${window}-day trailing net` : 'Net sentiment'}:{' '}
          <strong>{formatDecimalDisplay(d?.rolling_net_sentiment, 3)}</strong>
        </p>
        <p style={{ margin: 0, color: '#495057' }}>
          Mentions: <strong>{formatIntegerDisplay(d?.mention_count)}</strong>
        </p>
        {d?.returns != null && (
          <p
            style={{
              margin: '4px 0 0',
              color: d.returns >= 0 ? chartTheme.sentimentPos : chartTheme.sentimentNeg,
            }}
          >
            Daily return: <strong>{formatDecimalDisplay(d.returns * 100, 2)}%</strong>
          </p>
        )}
      </div>
    );
  };

  const sentimentLabel =
    window > 1 ? `Trailing net (${window}d)` : 'Net sentiment';

  return (
    <ResponsiveContainer width="100%" height={height} minWidth={0}>
      <ComposedChart
        data={chartData}
        margin={{ top: 16, right: 52, left: 4, bottom: 8 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} vertical={false} />
        <XAxis
          dataKey="shortDate"
          tick={{ fontSize: 11, fill: chartTheme.axis }}
          tickLine={false}
          axisLine={{ stroke: chartTheme.grid }}
          interval={tickEvery > 1 ? tickEvery - 1 : 'preserveStartEnd'}
          minTickGap={28}
        />
        <YAxis
          yAxisId="price"
          orientation="left"
          tick={{ fontSize: 11, fill: chartTheme.price }}
          tickLine={false}
          axisLine={{ stroke: chartTheme.grid }}
          tickFormatter={(v) => `$${formatIntegerDisplay(v)}`}
          width={56}
          domain={['auto', 'auto']}
          label={{
            value: 'Price',
            angle: -90,
            position: 'insideLeft',
            style: { fontSize: 11, fill: chartTheme.price, fontWeight: 600 },
          }}
        />
        {showVolume && (
          <YAxis
            yAxisId="volume"
            orientation="left"
            hide
            domain={[0, maxMentions]}
            allowDecimals={false}
          />
        )}
        <YAxis
          yAxisId="sentiment"
          orientation="right"
          domain={[-1, 1]}
          tick={{ fontSize: 11, fill: chartTheme.sentimentStroke }}
          tickLine={false}
          axisLine={{ stroke: chartTheme.grid }}
          tickFormatter={(v) => formatDecimalDisplay(v, 1)}
          width={44}
          label={{
            value: sentimentLabel,
            angle: 90,
            position: 'insideRight',
            style: { fontSize: 10, fill: chartTheme.sentimentStroke, fontWeight: 600 },
          }}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend
          wrapperStyle={{ fontSize: 12, paddingTop: 12 }}
          formatter={(value) => <span style={{ color: '#495057' }}>{value}</span>}
        />

        {showVolume && (
          <Bar
            yAxisId="volume"
            dataKey="mention_count"
            fill={chartTheme.volumeBar}
            name="Daily mentions"
            barSize={Math.min(14, Math.max(4, 480 / chartData.length))}
            radius={[3, 3, 0, 0]}
          />
        )}

        <Line
          yAxisId="price"
          type="monotone"
          dataKey="close"
          stroke={chartTheme.price}
          strokeWidth={2.5}
          dot={false}
          name="Close"
          activeDot={{ r: 5, stroke: '#fff', strokeWidth: 2 }}
        />

        {showRawOverlay && (
          <Line
            yAxisId="sentiment"
            type="monotone"
            dataKey="net_sentiment"
            stroke={chartTheme.axis}
            strokeWidth={1.25}
            strokeDasharray="4 4"
            strokeOpacity={0.65}
            dot={false}
            name="Same-day net"
          />
        )}

        <Area
          yAxisId="sentiment"
          type="monotone"
          dataKey="rolling_net_sentiment"
          stroke={chartTheme.sentimentStroke}
          fill={chartTheme.sentimentFill}
          fillOpacity={1}
          strokeWidth={2.5}
          name={window > 1 ? `${window}-day trailing net` : 'Net sentiment'}
          activeDot={{ r: 4 }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
};

export default SentimentPriceChart;
