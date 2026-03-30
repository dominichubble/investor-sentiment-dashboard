import React, { useEffect, useState } from 'react';
import { GlobalMarketSentimentChart } from '../../components/Charts';
import { ErrorBoundary } from '../../components/ErrorBoundary';
import Navbar from '../../components/Navbar';
import {
  apiService,
  type DailyTrendPoint,
  type SentimentSourceFilter,
  type SourceComparison,
  type StatisticsResponse,
} from '../../services/api';
import '../StockAnalysis/StockAnalysis.css';

const SOURCE_FILTER_OPTIONS: { value: SentimentSourceFilter; label: string }[] = [
  { value: 'all', label: 'All sources' },
  { value: 'reddit', label: 'Reddit' },
  { value: 'news', label: 'News' },
  { value: 'twitter', label: 'X' },
];

function SourceComparisonBars({ comp }: { comp: SourceComparison }) {
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
                {hasData ? `${b.total.toLocaleString()} records` : 'No data'}
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
    <div className="stock-analysis">
      <Navbar
        title="Market overview"
        subtitle="Aggregate sentiment across your ingested data: compare Reddit, news, and X, then inspect daily volume and net sentiment (not tied to a single ticker)."
      />

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
          <SourceComparisonBars comp={marketStats.source_comparison} />
        )}
        {sentimentSource !== 'all' && marketStats && (
          <div className="sa-source-filter-summary" aria-live="polite">
            <span>
              <strong>{marketStats.total_predictions.toLocaleString()}</strong> records
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
        {!marketStatsLoading && marketDailyTrend.length > 0 && (
          <ErrorBoundary fallbackTitle="Failed to render market-wide sentiment chart">
            <>
              <h4 className="sa-subsection-title">
                Daily volume &amp; net sentiment
                {sentimentSource !== 'all'
                  ? ` — ${SOURCE_FILTER_OPTIONS.find((o) => o.value === sentimentSource)?.label ?? sentimentSource}`
                  : ''}
              </h4>
              <p className="sa-section-desc sa-section-desc--tight">
                Stock-mention records in the last 90 days (relative to the newest record in range). Bars:
                mentions per day; line: net sentiment (positive minus negative, normalized by volume).
              </p>
              <GlobalMarketSentimentChart data={marketDailyTrend} height={300} />
            </>
          </ErrorBoundary>
        )}
        {!marketStatsLoading && marketDailyTrend.length === 0 && (
          <p className="sa-source-panel__empty">
            No daily trend data for this source and window. Ingest Reddit, news, or X data to populate charts.
          </p>
        )}
      </div>
    </div>
  );
};

export default MarketOverview;
