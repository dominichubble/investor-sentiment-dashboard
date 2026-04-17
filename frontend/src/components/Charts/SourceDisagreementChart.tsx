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
import type { SourceDisagreementDay } from '../../services/api';
import { formatDecimalDisplay, formatIntegerDisplay, paddedNiceCountAxisMax } from '../../utils/formatDisplay';
import { chartTheme } from './chartTheme';

interface SourceDisagreementChartProps {
  data: SourceDisagreementDay[];
  height?: number;
}

type Row = SourceDisagreementDay & { shortDate: string; fullDate: string };

const SourceDisagreementChart: React.FC<SourceDisagreementChartProps> = ({
  data,
  height = 320,
}) => {
  const { chartData, maxCount, maxSpread } = useMemo(() => {
    const rows: Row[] = (data ?? []).map((p) => ({
      ...p,
      shortDate: p.date.slice(5),
      fullDate: p.date,
    }));
    const maxC = Math.max(1, ...rows.map((r) => r.total_mentions || 0));
    const ranges = rows
      .map((r) => r.disagreement_range)
      .filter((v): v is number => v != null && !Number.isNaN(v));
    const maxR = ranges.length ? Math.max(...ranges) : 0;
    return {
      chartData: rows,
      maxCount: paddedNiceCountAxisMax(maxC, 1.08),
      maxSpread: Math.max(0.35, maxR * 1.15, 0.01),
    };
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
    const d = payload[0]?.payload as Row;
    const nets = d.net_by_source || {};
    const counts = d.counts_by_source || {};
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
        <p style={{ margin: '0 0 4px', color: '#475569' }}>
          Cross-source spread (range):{' '}
          <strong style={{ color: chartTheme.price }}>
            {d.disagreement_range != null ? formatDecimalDisplay(d.disagreement_range, 3) : '—'}
          </strong>
        </p>
        <p style={{ margin: '0 0 8px', color: '#64748b', fontSize: 12 }}>
          Std across nets:{' '}
          {d.disagreement_std != null ? formatDecimalDisplay(d.disagreement_std, 3) : '—'} · Channels in spread:{' '}
          {d.n_sources_active}
        </p>
        <p style={{ margin: '0 0 4px', color: '#64748b', fontSize: 12 }}>
          Mentions (all three channels): <strong>{formatIntegerDisplay(d.total_mentions)}</strong>
        </p>
        {Object.keys(nets).length > 0 && (
          <div style={{ marginTop: 8, fontSize: 12, color: '#475569', lineHeight: 1.5 }}>
            {(['reddit', 'news', 'twitter'] as const).map((ch) =>
              nets[ch] !== undefined ? (
                <div key={ch}>
                  {ch}: net {formatDecimalDisplay(nets[ch], 3)}
                  {counts[ch] != null ? ` (${formatIntegerDisplay(counts[ch])} rows)` : ''}
                </div>
              ) : null,
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <ResponsiveContainer width="100%" height={height} minWidth={0} debounce={120}>
      <ComposedChart data={chartData} margin={{ top: 12, right: 52, left: 4, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} vertical={false} />
        <XAxis
          dataKey="shortDate"
          tick={{ fontSize: 11, fill: chartTheme.axis }}
          tickLine={false}
          axisLine={{ stroke: chartTheme.grid }}
          interval={tickEvery > 0 ? tickEvery : 'preserveStartEnd'}
          minTickGap={24}
        />
        <YAxis yAxisId="vol" hide domain={[0, maxCount]} allowDecimals={false} />
        <YAxis
          yAxisId="spread"
          orientation="right"
          domain={[0, maxSpread]}
          tick={{ fontSize: 11, fill: chartTheme.price }}
          tickLine={false}
          width={44}
          tickFormatter={(v: number) => formatDecimalDisplay(v, 2)}
          label={{
            value: 'Spread',
            angle: -90,
            position: 'insideRight',
            style: { fill: chartTheme.price, fontSize: 11, fontWeight: 600 },
          }}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend
          wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
          formatter={(value) => <span style={{ color: '#475569' }}>{value}</span>}
        />
        <Bar
          yAxisId="vol"
          dataKey="total_mentions"
          name="Daily mentions (all channels)"
          fill="rgba(100, 116, 139, 0.35)"
          radius={[4, 4, 0, 0]}
          barSize={14}
        />
        <Line
          yAxisId="spread"
          type="monotone"
          dataKey="disagreement_range"
          name="Net sentiment range (max−min across channels)"
          stroke={chartTheme.price}
          strokeWidth={2}
          dot={{ r: 2, fill: chartTheme.price }}
          connectNulls={false}
          isAnimationActive={false}
        />
        <Line
          yAxisId="spread"
          type="monotone"
          dataKey="disagreement_std"
          name="Std of channel nets"
          stroke="#a855f7"
          strokeWidth={1.5}
          strokeDasharray="6 4"
          dot={false}
          connectNulls={false}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
};

export default SourceDisagreementChart;
