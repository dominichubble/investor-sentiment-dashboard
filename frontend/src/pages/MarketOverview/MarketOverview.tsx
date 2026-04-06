import React, { useEffect, useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { GlobalMarketSentimentChart, SourceDisagreementChart } from '../../components/Charts';
import { chartTheme } from '../../components/Charts/chartTheme';
import { ErrorBoundary } from '../../components/ErrorBoundary';
import Navbar from '../../components/Navbar';
import {
  apiService,
  type DailyTrendPoint,
  type SentimentSourceFilter,
  type SourceComparison,
  type StatisticsResponse,
  type StockInfo,
} from '../../services/api';
import { formatDecimalDisplay, formatIntegerDisplay } from '../../utils/formatDisplay';
import '../StockAnalysis/StockAnalysis.css';
import './MarketOverview.css';

const SOURCE_FILTER_OPTIONS: { value: SentimentSourceFilter; label: string }[] = [
  { value: 'all', label: 'All sources' },
  { value: 'reddit', label: 'Reddit' },
  { value: 'news', label: 'News' },
  { value: 'twitter', label: 'X' },
];

const PIE_COLORS = {
  Positive: chartTheme.sentimentPos,
  Neutral: '#868e96',
  Negative: chartTheme.sentimentNeg,
};

/** Recharts axis label styling */
const AXIS_LABEL_STYLE = { fill: '#868e96', fontSize: 11, fontWeight: 500 as const };

function ChartFootnote({ children }: { children: React.ReactNode }) {
  return <p className="mo-chart-footnote">{children}</p>;
}

function formatShortDate(iso: string | null | undefined): string {
  if (!iso) return '—';
  const d = iso.slice(0, 10);
  return d || '—';
}

function SourceComparisonBars({
  comp,
  dateSpanText,
  filterLabel,
}: {
  comp: SourceComparison;
  dateSpanText: string;
  filterLabel: string;
}) {
  const channels: { key: keyof SourceComparison; label: string }[] = [
    { key: 'reddit', label: 'Reddit' },
    { key: 'news', label: 'News' },
    { key: 'twitter', label: 'X (Twitter)' },
  ];
  return (
    <div className="sa-source-compare" role="group" aria-label="Sentiment mix by source">
      {channels.map(({ key, label }) => {
        const b = comp[key];
        const hasData = b && b.total > 0;
        return (
          <div key={key} className="sa-source-compare__card">
            <div className="sa-source-compare__head">
              <span className="sa-source-compare__title">{label}</span>
              <span className="sa-source-compare__n">
                {hasData ? `${formatIntegerDisplay(b.total)} records` : 'No data'}
              </span>
            </div>
            {hasData ? (
              <>
                <div className="sa-source-compare__bar-wrap" aria-hidden>
                  <div className="sa-source-compare__bar">
                    <div
                      className="sa-source-compare__seg sa-source-compare__seg--pos"
                      style={{ width: `${b.positive_percentage}%` }}
                    />
                    <div
                      className="sa-source-compare__seg sa-source-compare__seg--neu"
                      style={{ width: `${b.neutral_percentage}%` }}
                    />
                    <div
                      className="sa-source-compare__seg sa-source-compare__seg--neg"
                      style={{ width: `${b.negative_percentage}%` }}
                    />
                  </div>
                </div>
                <div className="sa-source-compare__legend">
                  <span className="sa-source-compare__legend-pos">
                    +{b.positive_percentage.toFixed(0)}%
                  </span>
                  <span className="sa-source-compare__legend-neu">
                    ~{b.neutral_percentage.toFixed(0)}%
                  </span>
                  <span className="sa-source-compare__legend-neg">
                    −{b.negative_percentage.toFixed(0)}%
                  </span>
                </div>
              </>
            ) : (
              <p className="sa-source-compare__empty">No ingested rows for this channel in range.</p>
            )}
          </div>
        );
      })}
      <ChartFootnote>
        <strong>Window:</strong> up to 90 days of ingested data ending at the newest record (
        {dateSpanText}). <strong>Source filter:</strong> {filterLabel}. Each channel counts FinBERT-labelled,
        stock-level rows in that window (same basis as the KPI strip).
      </ChartFootnote>
    </div>
  );
}

function MarketKpiStrip({
  stats,
  filterLabel,
  dateSpanText,
}: {
  stats: StatisticsResponse;
  filterLabel: string;
  dateSpanText: string;
}) {
  const { recent_activity, date_range, total_predictions, total_stocks_analyzed } = stats;
  return (
    <div className="mo-kpi-grid" role="region" aria-label="Key database statistics">
      <div className="mo-kpi-card mo-kpi-card--span">
        <div className="mo-kpi-value" style={{ fontSize: '0.95rem', fontWeight: 600, lineHeight: 1.4 }}>
          {dateSpanText} · {filterLabel}
        </div>
        <div className="mo-kpi-label">Statistics window</div>
        <div className="mo-kpi-sub">
          Rolling cap of 90 days before the newest record; all KPIs and charts below use this window unless
          noted.
        </div>
      </div>
      <div className="mo-kpi-card">
        <div className="mo-kpi-value">{formatIntegerDisplay(total_predictions)}</div>
        <div className="mo-kpi-label">Stock-level records</div>
        <div className="mo-kpi-sub">FinBERT-classified rows in this window</div>
      </div>
      <div className="mo-kpi-card">
        <div className="mo-kpi-value">{formatIntegerDisplay(total_stocks_analyzed)}</div>
        <div className="mo-kpi-label">Distinct tickers</div>
        <div className="mo-kpi-sub">At least one row in window</div>
      </div>
      <div className="mo-kpi-card">
        <div className="mo-kpi-value">
          {formatShortDate(date_range.latest)}
        </div>
        <div className="mo-kpi-label">Newest record</div>
        <div className="mo-kpi-sub">Earliest in window: {formatShortDate(date_range.earliest)}</div>
      </div>
      <div className="mo-kpi-card">
        <div className="mo-kpi-value">{formatIntegerDisplay(recent_activity.last_24h)}</div>
        <div className="mo-kpi-label">Last 24h</div>
        <div className="mo-kpi-sub">Hours before newest timestamp (not “today” if data is old)</div>
      </div>
      <div className="mo-kpi-card">
        <div className="mo-kpi-value">{formatIntegerDisplay(recent_activity.last_7d)}</div>
        <div className="mo-kpi-label">Last 7 days</div>
        <div className="mo-kpi-sub">Relative to newest timestamp</div>
      </div>
      <div className="mo-kpi-card">
        <div className="mo-kpi-value">{formatIntegerDisplay(recent_activity.last_30d)}</div>
        <div className="mo-kpi-label">Last 30 days</div>
        <div className="mo-kpi-sub">Relative to newest timestamp</div>
      </div>
    </div>
  );
}

function SentimentPiePanel({
  stats,
  filterLabel,
  dateSpanText,
}: {
  stats: StatisticsResponse;
  filterLabel: string;
  dateSpanText: string;
}) {
  const d = stats.sentiment_distribution;
  const total = d.positive + d.neutral + d.negative;
  const data = [
    { name: 'Positive', value: d.positive, pct: d.positive_percentage },
    { name: 'Neutral', value: d.neutral, pct: d.neutral_percentage },
    { name: 'Negative', value: d.negative, pct: d.negative_percentage },
  ];

  if (total <= 0) {
    return (
      <div className="mo-chart-panel">
        <h4 className="mo-chart-panel__title">Sentiment mix</h4>
        <p className="mo-empty-hint">No labelled records in this filter.</p>
      </div>
    );
  }

  return (
    <div className="mo-chart-panel">
      <h4 className="mo-chart-panel__title">Sentiment mix</h4>
      <p className="mo-chart-panel__desc">
        Share of positive, neutral, and negative labels for the current source filter (same 90-day window).
      </p>
      <ResponsiveContainer width="100%" height={220} minWidth={0}>
        <PieChart>
          <Pie
            data={data}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="50%"
            innerRadius={58}
            outerRadius={82}
            paddingAngle={2}
          >
            {data.map((entry) => (
              <Cell
                key={entry.name}
                fill={PIE_COLORS[entry.name as keyof typeof PIE_COLORS]}
                stroke="var(--color-surface)"
                strokeWidth={2}
              />
            ))}
          </Pie>
          <Tooltip
            formatter={(value: number, _n, item: { payload?: { pct?: number } }) => [
              `${formatIntegerDisplay(value)} (${formatDecimalDisplay(item.payload?.pct ?? 0, 1)}%)`,
              'Labelled rows',
            ]}
            contentStyle={{
              borderRadius: 8,
              border: `1px solid ${chartTheme.tooltipBorder}`,
              fontSize: 13,
            }}
          />
          <Legend
            verticalAlign="bottom"
            formatter={(v) => <span style={{ color: 'var(--color-text-secondary)', fontSize: 12 }}>{v}</span>}
          />
        </PieChart>
      </ResponsiveContainer>
      <ChartFootnote>
        <strong>Window:</strong> {dateSpanText} (≤90 days before newest record). <strong>Filter:</strong>{' '}
        {filterLabel}. Slices count FinBERT sentiment labels on stock-level ingested rows in that window.
      </ChartFootnote>
    </div>
  );
}

function RecentActivityBars({
  stats,
  filterLabel,
  dateSpanText,
}: {
  stats: StatisticsResponse;
  filterLabel: string;
  dateSpanText: string;
}) {
  const { recent_activity } = stats;
  const data = [
    { label: '24 hours', n: recent_activity.last_24h },
    { label: '7 days', n: recent_activity.last_7d },
    { label: '30 days', n: recent_activity.last_30d },
  ];
  const maxN = Math.max(1, ...data.map((x) => x.n));

  return (
    <div className="mo-chart-panel">
      <h4 className="mo-chart-panel__title">Activity by window</h4>
      <p className="mo-chart-panel__desc">
        Record counts relative to the latest timestamp in the dataset (not calendar “today” if data is historical).
      </p>
      <ResponsiveContainer width="100%" height={240} minWidth={0}>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16, bottom: 28, top: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} horizontal={false} />
          <XAxis
            type="number"
            domain={[0, maxN * 1.08]}
            tick={{ fontSize: 11, fill: chartTheme.axis }}
            label={{
              value: 'Record count',
              position: 'insideBottom',
              offset: -18,
              style: AXIS_LABEL_STYLE,
            }}
          />
          <YAxis
            type="category"
            dataKey="label"
            width={80}
            tick={{ fontSize: 11, fill: chartTheme.axis }}
            axisLine={false}
            tickLine={false}
            label={{
              value: 'Lookback from newest DB time',
              angle: -90,
              position: 'insideLeft',
              style: AXIS_LABEL_STYLE,
              offset: 8,
            }}
          />
          <Bar dataKey="n" fill={chartTheme.rollingLine} radius={[0, 6, 6, 0]} name="Records" />
          <Tooltip
            formatter={(v: number) => [formatIntegerDisplay(v), 'Records']}
            contentStyle={{
              borderRadius: 8,
              border: `1px solid ${chartTheme.tooltipBorder}`,
              fontSize: 13,
            }}
          />
        </BarChart>
      </ResponsiveContainer>
      <ChartFootnote>
        Bars are <strong>how many rows fall in each trailing period</strong>, measured backward from the
        dataset’s newest timestamp ({formatShortDate(stats.date_range.latest)}), not from today’s clock.
        Same source filter as elsewhere: {filterLabel}. Overall window: {dateSpanText}.
      </ChartFootnote>
    </div>
  );
}

function SourceVolumePanel({
  comp,
  dateSpanText,
}: {
  comp: SourceComparison;
  dateSpanText: string;
}) {
  const data = [
    { channel: 'Reddit', total: comp.reddit.total },
    { channel: 'News', total: comp.news.total },
    { channel: 'X', total: comp.twitter.total },
  ];
  const anyData = data.some((x) => x.total > 0);
  if (!anyData) {
    return (
      <div className="mo-chart-panel mo-full-width">
        <h4 className="mo-chart-panel__title">Volume by channel</h4>
        <p className="mo-empty-hint">No channel totals in range.</p>
      </div>
    );
  }
  const maxV = Math.max(1, ...data.map((x) => x.total));

  return (
    <div className="mo-chart-panel mo-full-width">
      <h4 className="mo-chart-panel__title">Volume by channel</h4>
      <p className="mo-chart-panel__desc">
        Total stock-level records per ingest platform (all sources view, same 90-day window).
      </p>
      <ResponsiveContainer width="100%" height={220} minWidth={0}>
        <BarChart data={data} margin={{ top: 8, right: 16, left: 8, bottom: 36 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} vertical={false} />
          <XAxis
            dataKey="channel"
            tick={{ fontSize: 12, fill: chartTheme.axis }}
            tickLine={false}
            axisLine={false}
            label={{ value: 'Ingest channel', position: 'insideBottom', offset: -22, style: AXIS_LABEL_STYLE }}
          />
          <YAxis
            domain={[0, maxV * 1.1]}
            tick={{ fontSize: 11, fill: chartTheme.axis }}
            label={{
              value: 'Stock-level rows',
              angle: -90,
              position: 'insideLeft',
              style: AXIS_LABEL_STYLE,
              offset: 2,
            }}
          />
          <Bar dataKey="total" fill="#5c7cfa" radius={[6, 6, 0, 0]} name="Records" />
          <Tooltip
            formatter={(v: number) => [v.toLocaleString(), 'Records']}
            contentStyle={{
              borderRadius: 8,
              border: `1px solid ${chartTheme.tooltipBorder}`,
              fontSize: 13,
            }}
          />
        </BarChart>
      </ResponsiveContainer>
      <ChartFootnote>
        Totals are <strong>all sources combined</strong> (this chart only appears in “All sources” view).
        Each bar is FinBERT-labelled stock-level rows from that channel in the same window: {dateSpanText}.
      </ChartFootnote>
    </div>
  );
}

function TopStocksPanel({
  stocks,
  filterLabel,
  dateSpanText,
}: {
  stocks: StockInfo[];
  filterLabel: string;
  dateSpanText: string;
}) {
  const rows = useMemo(() => stocks.slice(0, 12).filter((s) => s.count > 0), [stocks]);
  if (rows.length === 0) {
    return (
      <div className="mo-chart-panel mo-full-width mo-top-stocks">
        <h4 className="mo-chart-panel__title">Most-mentioned tickers</h4>
        <p className="mo-empty-hint">No ticker aggregates in this window.</p>
      </div>
    );
  }
  const maxC = Math.max(1, ...rows.map((r) => r.count));

  return (
    <div className="mo-chart-panel mo-full-width mo-top-stocks">
      <h4 className="mo-chart-panel__title">Most-mentioned tickers</h4>
      <p className="mo-chart-panel__desc">
        Ranked by how many stock-level ingested rows mention each ticker (same statistics API as the KPIs).
        Colours hint sentiment tilt: greener = higher positive share, redder = higher negative share.
      </p>
      <ResponsiveContainer width="100%" height={Math.max(260, rows.length * 36)} minWidth={0}>
        <BarChart data={rows} layout="vertical" margin={{ left: 4, right: 24, bottom: 28, top: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} horizontal={false} />
          <XAxis
            type="number"
            domain={[0, maxC * 1.08]}
            tick={{ fontSize: 11, fill: chartTheme.axis }}
            tickFormatter={(v: number) => formatIntegerDisplay(v)}
            label={{
              value: 'Rows mentioning ticker',
              position: 'insideBottom',
              offset: -18,
              style: AXIS_LABEL_STYLE,
            }}
          />
          <YAxis
            type="category"
            dataKey="ticker"
            width={56}
            tick={{ fontSize: 12, fill: chartTheme.axis, fontWeight: 600 }}
            axisLine={false}
            tickLine={false}
            label={{
              value: 'Ticker',
              angle: -90,
              position: 'insideLeft',
              style: AXIS_LABEL_STYLE,
              offset: 4,
            }}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const p = payload[0].payload as StockInfo;
              return (
                <div
                  style={{
                    background: chartTheme.tooltipBg,
                    border: `1px solid ${chartTheme.tooltipBorder}`,
                    borderRadius: 8,
                    padding: '10px 14px',
                    fontSize: 13,
                  }}
                >
                  <div style={{ fontWeight: 700, marginBottom: 6 }}>{p.ticker}</div>
                  <div style={{ color: '#64748b' }}>{p.company_name}</div>
                  <div style={{ marginTop: 6 }}>
                    Rows (window): <strong>{formatIntegerDisplay(p.count)}</strong>
                  </div>
                  <div style={{ fontSize: 12, marginTop: 4, whiteSpace: 'nowrap' }}>
                    +{formatIntegerDisplay(p.positive)} / ~{formatIntegerDisplay(p.neutral)} / −
                    {formatIntegerDisplay(p.negative)}
                  </div>
                </div>
              );
            }}
          />
          <Bar dataKey="count" radius={[0, 6, 6, 0]} name="Rows">
            {rows.map((row) => {
              const posR = row.count > 0 ? row.positive / row.count : 0;
              const negR = row.count > 0 ? row.negative / row.count : 0;
              let fill = chartTheme.axis;
              if (posR > negR + 0.08) fill = chartTheme.sentimentPos;
              else if (negR > posR + 0.08) fill = chartTheme.sentimentNeg;
              return <Cell key={row.ticker} fill={fill} />;
            })}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <ChartFootnote>
        <strong>Time frame:</strong> {dateSpanText} (≤90 days ending at newest record). <strong>Source filter:</strong>{' '}
        {filterLabel}. “Mentions” are ingested rows tied to that ticker after ticker detection — not raw post
        counts.
      </ChartFootnote>
    </div>
  );
}

const MarketOverview: React.FC = () => {
  const [marketStats, setMarketStats] = useState<StatisticsResponse | null>(null);
  const [marketStatsLoading, setMarketStatsLoading] = useState(true);
  const [sentimentSource, setSentimentSource] = useState<SentimentSourceFilter>('all');

  const marketDailyTrend: DailyTrendPoint[] = Array.isArray(marketStats?.daily_trend)
    ? marketStats!.daily_trend
    : [];

  const filterLabel = useMemo(
    () => SOURCE_FILTER_OPTIONS.find((o) => o.value === sentimentSource)?.label ?? 'All sources',
    [sentimentSource],
  );

  const dateSpanText = useMemo(() => {
    if (!marketStats?.date_range) return '—';
    return `${formatShortDate(marketStats.date_range.earliest)} → ${formatShortDate(marketStats.date_range.latest)}`;
  }, [marketStats]);

  const dailyCalendarSpanText = useMemo(() => {
    if (!marketDailyTrend.length) return '';
    const dates = [...new Set(marketDailyTrend.map((d) => d.date))].sort();
    const lo = dates[0];
    const hi = dates[dates.length - 1];
    return lo === hi ? lo : `${lo} → ${hi}`;
  }, [marketDailyTrend]);

  useEffect(() => {
    let cancelled = false;
    setMarketStatsLoading(true);
    apiService
      .getStatistics({
        days: 90,
        data_source: sentimentSource === 'all' ? undefined : sentimentSource,
      })
      .then((s) => {
        if (!cancelled) setMarketStats(s);
      })
      .catch(() => {
        if (!cancelled) setMarketStats(null);
      })
      .finally(() => {
        if (!cancelled) setMarketStatsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [sentimentSource]);

  return (
    <div id="main-content" className="stock-analysis" tabIndex={-1}>
      <Navbar
        title="Market overview"
        subtitle="Aggregate sentiment across your ingested data: compare Reddit, news, and X, then inspect daily volume and net sentiment (not tied to a single ticker)."
      />

      {!marketStatsLoading && marketStats && (
        <MarketKpiStrip stats={marketStats} filterLabel={filterLabel} dateSpanText={dateSpanText} />
      )}

      <div className="sa-chart-section sa-market-wide sa-source-panel">
        <h3 className="sa-section-title">Sentiment by data source</h3>
        <p className="sa-section-desc">
          Compare how positive, neutral, and negative labels are distributed across Reddit, news, and X in
          your database. Use the toggle to filter the daily chart and headline counts to one channel, or view
          all sources together.
        </p>
        <div
          className="sa-source-filter"
          role="tablist"
          aria-label="Filter dashboard statistics by ingest source"
        >
          {SOURCE_FILTER_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              type="button"
              role="tab"
              aria-selected={sentimentSource === opt.value}
              className={`sa-source-filter__btn ${sentimentSource === opt.value ? 'is-active' : ''}`}
              onClick={() => setSentimentSource(opt.value)}
            >
              {opt.label}
            </button>
          ))}
        </div>
        {sentimentSource === 'all' && marketStats?.source_comparison && (
          <SourceComparisonBars
            comp={marketStats.source_comparison}
            dateSpanText={dateSpanText}
            filterLabel={filterLabel}
          />
        )}
        {sentimentSource !== 'all' && marketStats && (
          <div className="sa-source-filter-summary" aria-live="polite">
            <span>
              <strong>{formatIntegerDisplay(marketStats.total_predictions)}</strong> records
            </span>
            <span className="sa-source-filter-summary__sep">·</span>
            <span>
              Net mix:{' '}
              <span className="sa-source-compare__legend-pos">
                +{marketStats.sentiment_distribution.positive_percentage.toFixed(0)}%
              </span>
              {' / '}
              <span className="sa-source-compare__legend-neu">
                ~{marketStats.sentiment_distribution.neutral_percentage.toFixed(0)}%
              </span>
              {' / '}
              <span className="sa-source-compare__legend-neg">
                −{marketStats.sentiment_distribution.negative_percentage.toFixed(0)}%
              </span>
            </span>
          </div>
        )}
        {marketStatsLoading && (
          <p className="sa-source-panel__loading">Loading source statistics…</p>
        )}
      </div>

      {!marketStatsLoading && marketStats && (
        <ErrorBoundary fallbackTitle="Failed to render overview charts">
          <div className="mo-charts-grid">
            <SentimentPiePanel
              stats={marketStats}
              filterLabel={filterLabel}
              dateSpanText={dateSpanText}
            />
            <RecentActivityBars
              stats={marketStats}
              filterLabel={filterLabel}
              dateSpanText={dateSpanText}
            />
          </div>
          {sentimentSource === 'all' && marketStats.source_comparison && (
            <SourceVolumePanel comp={marketStats.source_comparison} dateSpanText={dateSpanText} />
          )}
          {sentimentSource === 'all' &&
            marketStats.source_disagreement_trend &&
            marketStats.source_disagreement_trend.length > 0 && (
              <ErrorBoundary fallbackTitle="Failed to render cross-source disagreement chart">
                <div className="sa-chart-section sa-market-wide sa-source-panel mo-full-width">
                  <h4 className="sa-subsection-title">Cross-source disagreement over time</h4>
                  <p className="sa-section-desc sa-section-desc--tight">
                    Each day we compute <strong>net sentiment</strong> per channel (positive minus negative, divided
                    by volume) for Reddit, news, and X separately. A channel is included only if it has at least{' '}
                    <strong>three</strong> labelled rows that day. The <strong>purple line</strong> is the spread:
                    maximum net minus minimum net across qualifying channels — high values mean platforms disagree on
                    mood. The dashed line is the standard deviation of those channel nets. Grey bars are total
                    mentions that day across all three channels. Same statistics window as above (
                    <strong>{dateSpanText}</strong>).
                  </p>
                  <SourceDisagreementChart
                    data={marketStats.source_disagreement_trend}
                    height={320}
                  />
                  <ChartFootnote>
                    This view is only available for <strong>All sources</strong>. It is descriptive, not a trading
                    signal; thin days or missing channels reduce or blank the spread metrics.
                  </ChartFootnote>
                </div>
              </ErrorBoundary>
            )}
          <TopStocksPanel
            stocks={marketStats.top_stocks}
            filterLabel={filterLabel}
            dateSpanText={dateSpanText}
          />
        </ErrorBoundary>
      )}

      {!marketStatsLoading && marketDailyTrend.length > 0 && (
        <ErrorBoundary fallbackTitle="Failed to render market-wide sentiment chart">
          <div className="sa-chart-section sa-market-wide sa-source-panel">
            <h4 className="sa-subsection-title">
              Daily volume &amp; net sentiment
              {sentimentSource !== 'all'
                ? ` — ${SOURCE_FILTER_OPTIONS.find((o) => o.value === sentimentSource)?.label ?? sentimentSource}`
                : ''}
            </h4>
            <p className="sa-section-desc sa-section-desc--tight">
              One point per calendar day that has data. Bars: stock-level rows that day; line: net sentiment
              (positive minus negative, scaled by volume). The horizontal span shown is{' '}
              <strong>{dailyCalendarSpanText}</strong> — within the overall stats window{' '}
              <strong>{dateSpanText}</strong> and source filter <strong>{filterLabel}</strong>.
            </p>
            <GlobalMarketSentimentChart data={marketDailyTrend} height={300} />
            <ChartFootnote>
              Daily buckets use the same ingested timestamps as the rest of the dashboard. Days with zero rows
              are omitted from the series, so the chart may not cover every calendar day in the window.
            </ChartFootnote>
          </div>
        </ErrorBoundary>
      )}
      {!marketStatsLoading && marketStats && marketDailyTrend.length === 0 && (
        <p className="sa-source-panel__empty">
          No daily trend data for this source and window. Ingest Reddit, news, or X data to populate charts.
        </p>
      )}
    </div>
  );
};

export default MarketOverview;
