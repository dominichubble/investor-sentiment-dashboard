import React, { useState, useEffect, useCallback, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { useSearchParams } from 'react-router-dom';
import {
  SentimentPriceChart,
  DailySentimentAggregateChart,
  CorrelationScatter,
  LagChart,
  RollingCorrelationChart,
} from '../../components/Charts';
import { ErrorBoundary } from '../../components/ErrorBoundary';
import Navbar from '../../components/Navbar';
import { apiService, type TickerNarrativeResponse } from '../../services/api';
import type {
  CorrelationResponse,
  LagAnalysisResponse,
  TimeSeriesResponse,
  StockInfoResponse,
  GrangerCausalityResponse,
  RollingCorrelationResponse,
} from '../../types';
import './StockAnalysis.css';

const PERIOD_OPTIONS = [
  { label: '30 Days', value: '30d' },
  { label: '90 Days', value: '90d' },
  { label: '6 Months', value: '6mo' },
  { label: '1 Year', value: '1y' },
  { label: 'Custom', value: 'custom' },
];

function toISODate(d: Date): string {
  return d.toISOString().slice(0, 10);
}

function defaultCustomRange(): { start: string; end: string } {
  const end = new Date();
  const start = new Date();
  start.setDate(start.getDate() - 90);
  return { start: toISODate(start), end: toISODate(end) };
}

type CustomRangePayload = { start_date: string; end_date: string } | null;

const StockAnalysis: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const didRunUrlInit = useRef(false);

  const [ticker, setTicker] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const [period, setPeriod] = useState('90d');
  const [dateRangeMode, setDateRangeMode] = useState<'preset' | 'custom'>('preset');
  const [customStart, setCustomStart] = useState(() => defaultCustomRange().start);
  const [customEnd, setCustomEnd] = useState(() => defaultCustomRange().end);

  // Data states
  const [correlation, setCorrelation] = useState<CorrelationResponse | null>(null);
  const [timeSeries, setTimeSeries] = useState<TimeSeriesResponse | null>(null);
  const [lagAnalysis, setLagAnalysis] = useState<LagAnalysisResponse | null>(null);
  const [grangerData, setGrangerData] = useState<GrangerCausalityResponse | null>(null);
  const [rollingData, setRollingData] = useState<RollingCorrelationResponse | null>(null);
  const [stockInfo, setStockInfo] = useState<StockInfoResponse | null>(null);

  // UI states
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  /** Trailing days for net sentiment when comparing to price (causal moving average). */
  const [sentimentMemoryDays, setSentimentMemoryDays] = useState(3);

  const [tickerNarrative, setTickerNarrative] = useState<TickerNarrativeResponse | null>(null);
  const [tickerNarrativeLoading, setTickerNarrativeLoading] = useState(false);
  const [tickerNarrativeError, setTickerNarrativeError] = useState<string | null>(null);

  const analyzeStock = useCallback(
    async (
      stockTicker: string,
      analysisPeriod: string,
      customRange: CustomRangePayload,
      trailingDaysOverride?: number,
    ) => {
    if (!stockTicker) return;

    setIsLoading(true);
    setError(null);
    setTicker(stockTicker);

    const trailing_days = trailingDaysOverride ?? sentimentMemoryDays;
    const rangeParams = customRange
      ? {
          period: analysisPeriod,
          start_date: customRange.start_date,
          end_date: customRange.end_date,
          trailing_days,
        }
      : { period: analysisPeriod, trailing_days };

    try {
      const [corrData, tsData, lagData, infoData, grangerResult, rollingResult] = await Promise.allSettled([
        apiService.getCorrelation(stockTicker, rangeParams),
        apiService.getCorrelationTimeseries(stockTicker, rangeParams),
        apiService.getLagAnalysis(stockTicker, { max_lag_days: 5, ...rangeParams }),
        apiService.getStockInfo(stockTicker),
        apiService.getGrangerCausality(stockTicker, { max_lag: 5, ...rangeParams }),
        apiService.getRollingCorrelation(stockTicker, { window: 14, ...rangeParams }),
      ]);

      if (corrData.status === 'fulfilled') setCorrelation(corrData.value);
      else setCorrelation(null);

      if (tsData.status === 'fulfilled') setTimeSeries(tsData.value);
      else setTimeSeries(null);

      if (lagData.status === 'fulfilled') setLagAnalysis(lagData.value);
      else setLagAnalysis(null);

      if (infoData.status === 'fulfilled') setStockInfo(infoData.value);
      else setStockInfo(null);

      if (grangerResult.status === 'fulfilled') setGrangerData(grangerResult.value);
      else setGrangerData(null);

      if (rollingResult.status === 'fulfilled') setRollingData(rollingResult.value);
      else setRollingData(null);

      const allFailed = [corrData, tsData, lagData].every(r => r.status === 'rejected');
      if (allFailed) {
        setError('Could not fetch data for this ticker. Please check the ticker symbol and ensure the backend is running.');
      }

    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred');
    } finally {
      setIsLoading(false);
    }
  }, [sentimentMemoryDays]);

  useEffect(() => {
    setTickerNarrative(null);
    setTickerNarrativeError(null);
  }, [ticker, period, dateRangeMode, customStart, customEnd]);

  const fetchTickerNarrative = useCallback(
    async (forceRefresh: boolean) => {
      if (!ticker) return;
      setTickerNarrativeLoading(true);
      setTickerNarrativeError(null);
      try {
        const params =
          dateRangeMode === 'custom' && customStart && customEnd
            ? {
                period,
                start_date: customStart,
                end_date: customEnd,
                force_refresh: forceRefresh,
              }
            : { period, force_refresh: forceRefresh };
        const data = await apiService.getTickerSentimentNarrative(ticker, params);
        if (data.error) {
          setTickerNarrative(null);
          setTickerNarrativeError(data.error);
          return;
        }
        setTickerNarrative(data);
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : 'Failed to load narrative';
        setTickerNarrative(null);
        setTickerNarrativeError(msg);
      } finally {
        setTickerNarrativeLoading(false);
      }
    },
    [ticker, period, dateRangeMode, customStart, customEnd],
  );

  const syncUrl = (sym: string, p: string, custom: CustomRangePayload) => {
    if (custom) {
      setSearchParams({ ticker: sym, start: custom.start_date, end: custom.end_date });
    } else {
      setSearchParams({ ticker: sym, period: p });
    }
  };

  // Load ticker (and optional range) from URL on first mount
  useEffect(() => {
    if (didRunUrlInit.current) return;
    didRunUrlInit.current = true;

    const raw = searchParams.get('ticker')?.trim();
    const start = searchParams.get('start')?.trim();
    const end = searchParams.get('end')?.trim();
    const urlPeriod = searchParams.get('period')?.trim();

    const effectivePeriod =
      urlPeriod && PERIOD_OPTIONS.some((o) => o.value === urlPeriod) ? urlPeriod : '90d';
    if (urlPeriod && PERIOD_OPTIONS.some((o) => o.value === urlPeriod)) {
      setPeriod(urlPeriod);
    }

    if (start && end) {
      setDateRangeMode('custom');
      setCustomStart(start);
      setCustomEnd(end);
    }

    if (raw) {
      const sym = raw.toUpperCase();
      setSearchInput(sym);
      const custom: CustomRangePayload =
        start && end ? { start_date: start, end_date: end } : null;
      analyzeStock(sym, effectivePeriod, custom);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- intentional: run once on mount
  }, []);

  const validateCustomRange = (): string | null => {
    if (!customStart || !customEnd) {
      return 'Select both start and end dates for a custom range.';
    }
    if (customStart > customEnd) {
      return 'Start date must be on or before end date.';
    }
    return null;
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchInput.trim()) return;

    const sym = searchInput.trim().toUpperCase();
    if (dateRangeMode === 'custom') {
      const msg = validateCustomRange();
      if (msg) {
        setError(msg);
        return;
      }
      const custom: CustomRangePayload = {
        start_date: customStart,
        end_date: customEnd,
      };
      syncUrl(sym, period, custom);
      analyzeStock(sym, period, custom);
      return;
    }

    syncUrl(sym, period, null);
    analyzeStock(sym, period, null);
  };

  const handlePeriodChange = (newPeriod: string) => {
    setDateRangeMode('preset');
    setPeriod(newPeriod);
    if (ticker) {
      setSearchParams({ ticker, period: newPeriod });
      analyzeStock(ticker, newPeriod, null);
    }
  };

  const handleCustomRangeApply = () => {
    const msg = validateCustomRange();
    if (msg) {
      setError(msg);
      return;
    }
    setError(null);
    if (!ticker) return;
    const custom: CustomRangePayload = {
      start_date: customStart,
      end_date: customEnd,
    };
    syncUrl(ticker, period, custom);
    analyzeStock(ticker, period, custom);
  };

  const todayISO = toISODate(new Date());

  const getCorrelationColor = (r: number | undefined): string => {
    if (r === undefined) return 'var(--color-text-muted)';
    if (r > 0.3) return 'var(--color-positive)';
    if (r < -0.3) return 'var(--color-negative)';
    return 'var(--color-text-muted)';
  };

  const formatMarketCap = (cap: number | undefined): string => {
    if (!cap) return 'N/A';
    if (cap >= 1e12) return `$${(cap / 1e12).toFixed(1)}T`;
    if (cap >= 1e9) return `$${(cap / 1e9).toFixed(1)}B`;
    if (cap >= 1e6) return `$${(cap / 1e6).toFixed(1)}M`;
    return `$${cap.toLocaleString()}`;
  };

  return (
    <div className="stock-analysis">
      <Navbar
        title="Sentiment–price correlation"
        subtitle="Choose a preset lookback or a custom calendar range. All charts and statistics use the same window. For market-wide sentiment by source, use Market overview."
      />

      {/* Header / Search */}
      <div className="sa-header">
        <h1 className="sa-title">Analyze a stock</h1>
        <p className="sa-subtitle">
          Enter a ticker, pick how far back to look, then run the analysis. Custom ranges use inclusive start and end dates.
        </p>

        <form className="sa-search-form" onSubmit={handleSearch}>
          <input
            type="text"
            className="sa-search-input"
            placeholder="Enter stock ticker (e.g. AAPL, TSLA, MSFT)"
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value.toUpperCase())}
          />
          <button type="submit" className="sa-search-btn" disabled={isLoading || !searchInput.trim()}>
            {isLoading ? 'Analyzing...' : 'Analyze'}
          </button>
        </form>

        <div className="sa-range-mode" role="radiogroup" aria-label="Time window type">
          <label className="sa-range-mode-option">
            <input
              type="radio"
              name="rangeMode"
              checked={dateRangeMode === 'preset'}
              onChange={() => setDateRangeMode('preset')}
            />
            <span>Preset period</span>
          </label>
          <label className="sa-range-mode-option">
            <input
              type="radio"
              name="rangeMode"
              checked={dateRangeMode === 'custom'}
              onChange={() => setDateRangeMode('custom')}
            />
            <span>Custom date range</span>
          </label>
        </div>

        {dateRangeMode === 'preset' && (
          <div className="sa-period-selector">
            {PERIOD_OPTIONS.map(opt => (
              <button
                key={opt.value}
                type="button"
                className={`sa-period-btn ${period === opt.value ? 'active' : ''}`}
                onClick={() => handlePeriodChange(opt.value)}
              >
                {opt.label}
              </button>
            ))}
          </div>
        )}

        {dateRangeMode === 'custom' && (
          <div className="sa-custom-dates">
            <label className="sa-date-field">
              <span className="sa-date-label">From</span>
              <input
                type="date"
                className="sa-date-input"
                value={customStart}
                max={customEnd || todayISO}
                onChange={(e) => setCustomStart(e.target.value)}
              />
            </label>
            <label className="sa-date-field">
              <span className="sa-date-label">To</span>
              <input
                type="date"
                className="sa-date-input"
                value={customEnd}
                min={customStart}
                max={todayISO}
                onChange={(e) => setCustomEnd(e.target.value)}
              />
            </label>
            {ticker && (
              <button
                type="button"
                className="sa-apply-range-btn"
                disabled={isLoading}
                onClick={handleCustomRangeApply}
              >
                Apply range
              </button>
            )}
          </div>
        )}

        <div className="sa-memory-panel" role="region" aria-labelledby="sa-memory-title">
          <div className="sa-memory-panel__intro">
            <span className="sa-memory-panel__kicker">Sentiment smoothing</span>
            <h2 className="sa-memory-panel__title" id="sa-memory-title">
              Trailing net sentiment
            </h2>
            <p className="sa-memory-panel__lede">
              Choose how many <strong>past</strong> calendar days are averaged into each day’s net sentiment (no
              lookahead). This single control updates Pearson/Spearman, lag, Granger, rolling correlation, and the
              charts below.
            </p>
          </div>
          <div className="sa-memory-panel__control">
            <span className="sa-memory-panel__control-label" id="sa-memory-segment-label">
              Window length
            </span>
            <div
              className="sa-memory-segmented"
              role="group"
              aria-labelledby="sa-memory-segment-label"
            >
              {([1, 3, 5, 7] as const).map((d) => (
                <button
                  key={d}
                  type="button"
                  className={`sa-memory-segmented__btn ${sentimentMemoryDays === d ? 'is-active' : ''}`}
                  aria-pressed={sentimentMemoryDays === d}
                  onClick={() => {
                    setSentimentMemoryDays(d);
                    if (!ticker) return;
                    const custom: CustomRangePayload =
                      dateRangeMode === 'custom' && customStart && customEnd
                        ? { start_date: customStart, end_date: customEnd }
                        : null;
                    analyzeStock(ticker, period, custom, d);
                  }}
                >
                  <span className="sa-memory-segmented__value">{d === 1 ? '1' : d}</span>
                  <span className="sa-memory-segmented__unit">{d === 1 ? 'day' : 'days'}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="sa-error">
          <p>{error}</p>
        </div>
      )}

      {/* Loading */}
      {isLoading && (
        <div className="sa-loading">
          <div className="sa-spinner" />
          <p>Fetching price data and calculating correlations for {ticker}...</p>
        </div>
      )}

      {!isLoading && ticker && !correlation && error && (
        <div className="sa-results sa-results-partial">
          <p className="sa-partial-message">{error}</p>
        </div>
      )}

      {/* Results */}
      {!isLoading && ticker && correlation && (
        <div className="sa-results">
          {/* Stock Info Bar */}
          <div className="sa-info-bar">
            <div className="sa-info-left">
              <h2 className="sa-ticker-title">{ticker}</h2>
              {stockInfo && (
                <span className="sa-stock-name">{stockInfo.name}</span>
              )}
            </div>
            <div className="sa-info-right">
              {stockInfo?.sector && (
                <span className="sa-tag">{stockInfo.sector}</span>
              )}
              {stockInfo?.market_cap && (
                <span className="sa-tag">{formatMarketCap(stockInfo.market_cap)}</span>
              )}
              <span className="sa-tag">{correlation.data_points} data points</span>
            </div>
          </div>

          {/* Correlation Summary Cards */}
          <div className="sa-summary-grid">
            <div className="sa-stat-card">
              <div className="sa-stat-label">Pearson Correlation</div>
              <div
                className="sa-stat-value"
                style={{ color: getCorrelationColor(correlation.pearson?.coefficient) }}
              >
                {correlation.pearson
                  ? `${correlation.pearson.coefficient >= 0 ? '+' : ''}${correlation.pearson.coefficient.toFixed(4)}`
                  : 'N/A'}
              </div>
              <div className="sa-stat-detail">
                {correlation.pearson?.interpretation || 'Insufficient data'}
              </div>
              {correlation.pearson && (
                <div className={`sa-stat-badge ${correlation.pearson.significant ? 'significant' : 'not-significant'}`}>
                  {correlation.pearson.significant ? `p = ${correlation.pearson.p_value.toFixed(4)}` : 'Not significant'}
                </div>
              )}
            </div>

            <div className="sa-stat-card">
              <div className="sa-stat-label">Spearman Correlation</div>
              <div
                className="sa-stat-value"
                style={{ color: getCorrelationColor(correlation.spearman?.coefficient) }}
              >
                {correlation.spearman
                  ? `${correlation.spearman.coefficient >= 0 ? '+' : ''}${correlation.spearman.coefficient.toFixed(4)}`
                  : 'N/A'}
              </div>
              <div className="sa-stat-detail">
                {correlation.spearman?.interpretation || 'Insufficient data'}
              </div>
              {correlation.spearman && (
                <div className={`sa-stat-badge ${correlation.spearman.significant ? 'significant' : 'not-significant'}`}>
                  {correlation.spearman.significant ? `p = ${correlation.spearman.p_value.toFixed(4)}` : 'Not significant'}
                </div>
              )}
            </div>

            <div className="sa-stat-card">
              <div className="sa-stat-label">Best Lag</div>
              <div className="sa-stat-value" style={{ color: 'var(--color-accent)' }}>
                {lagAnalysis?.best_lag
                  ? lagAnalysis.best_lag.lag_days === 0
                    ? 'Same Day'
                    : lagAnalysis.best_lag.lag_days > 0
                      ? `+${lagAnalysis.best_lag.lag_days} Days`
                      : `${lagAnalysis.best_lag.lag_days} Days`
                  : 'N/A'}
              </div>
              <div className="sa-stat-detail">
                {lagAnalysis?.best_lag
                  ? `r = ${lagAnalysis.best_lag.pearson_r?.toFixed(4)}`
                  : 'No lag data'}
              </div>
              {lagAnalysis?.best_lag?.significant && (
                <div className="sa-stat-badge significant">
                  Sentiment {lagAnalysis.best_lag.lag_days > 0 ? 'leads' : 'follows'} price
                </div>
              )}
            </div>

            <div className="sa-stat-card">
              <div className="sa-stat-label">Signal Strength</div>
              <div className="sa-stat-value" style={{
                color: correlation.pearson?.significant ? 'var(--color-positive)' : 'var(--color-negative)'
              }}>
                {correlation.pearson?.significant ? 'Significant' : 'Weak'}
              </div>
              <div className="sa-stat-detail">
                {correlation.pearson?.significant
                  ? 'Statistical evidence of correlation'
                  : 'No statistically significant relationship detected'}
              </div>
            </div>
          </div>

          <div className="sa-chart-section sa-ticker-narrative">
            <h3 className="sa-section-title">AI sentiment narrative</h3>
            <p className="sa-section-desc">
              Summarises <strong>only ingested posts and news</strong> in the same date window as the
              charts above (via Groq&apos;s free API). Cached per ticker and data snapshot; add{' '}
              <code className="sa-inline-code">GROQ_API_KEY</code> to your backend <code className="sa-inline-code">.env</code>.
            </p>
            <div className="sa-ticker-narrative__actions">
              <button
                type="button"
                className="sa-narrative-btn"
                disabled={tickerNarrativeLoading}
                onClick={() => fetchTickerNarrative(false)}
              >
                {tickerNarrativeLoading ? 'Working…' : 'Get AI summary (uses cache if data unchanged)'}
              </button>
              <button
                type="button"
                className="sa-narrative-btn sa-narrative-btn--secondary"
                disabled={tickerNarrativeLoading}
                onClick={() => fetchTickerNarrative(true)}
              >
                Regenerate (new API call)
              </button>
            </div>
            {tickerNarrativeError && (
              <p className="sa-ticker-narrative__error" role="alert">
                {tickerNarrativeError}
              </p>
            )}
            {tickerNarrative && !tickerNarrativeError && (
              <div className="sa-ticker-narrative__meta">
                {tickerNarrative.cached ? (
                  <span className="sa-narrative-pill">Cached</span>
                ) : (
                  <span className="sa-narrative-pill">Fresh</span>
                )}
                {tickerNarrative.model && tickerNarrative.model !== 'none' && (
                  <span className="sa-narrative-pill">{tickerNarrative.model}</span>
                )}
                <span className="sa-narrative-pill">{tickerNarrative.record_count} mentions in window</span>
                {tickerNarrative.window_start && tickerNarrative.window_end && (
                  <span className="sa-narrative-pill">
                    {tickerNarrative.window_start.slice(0, 10)} → {tickerNarrative.window_end.slice(0, 10)}
                  </span>
                )}
              </div>
            )}
            {tickerNarrative?.narrative && (
              <div className="sa-ticker-narrative__body">
                <ReactMarkdown>{tickerNarrative.narrative}</ReactMarkdown>
              </div>
            )}
          </div>

          {/* Daily aggregated sentiment (mentions pooled by calendar day) */}
          {timeSeries && timeSeries.series.length > 0 && (
            <ErrorBoundary fallbackTitle="Failed to render daily sentiment chart">
              <div className="sa-chart-section">
                <h3 className="sa-section-title">Daily sentiment aggregate</h3>
                <p className="sa-section-desc">
                  Each day, all mentions of {ticker} are rolled into one bucket. The stacked bands show
                  the fraction of mentions classified positive, neutral, or negative; the line is net
                  sentiment (positive share minus negative share), from −1 to +1.
                </p>
                <DailySentimentAggregateChart data={timeSeries.series} height={400} />
              </div>
            </ErrorBoundary>
          )}

          {/* Dual-Axis Time Series — trailing sentiment vs price */}
          {timeSeries && timeSeries.series.length > 0 && (
            <ErrorBoundary fallbackTitle="Failed to render time series chart">
              <div className="sa-chart-section">
                <h3 className="sa-section-title">Sentiment vs price (with memory)</h3>
                <p className="sa-section-desc">
                  The <strong>filled</strong> series is a <strong>trailing average</strong> of daily net
                  sentiment: each point uses that calendar day and the previous N−1 days in range (no
                  lookahead). A run of positive days pulls the curve up before you compare it to{' '}
                  <strong>close</strong> (left axis). With N &gt; 1, the <strong>dashed</strong> line is
                  same-day net only. Bars are daily mention counts on their own scale. Change the trailing window
                  under the date range controls to refit correlations.
                </p>
                <SentimentPriceChart
                  data={timeSeries.series}
                  height={440}
                  rollingWindowDays={sentimentMemoryDays}
                  apiTrailingDays={timeSeries.trailing_days}
                />
              </div>
            </ErrorBoundary>
          )}

          {/* Scatter Plot & Lag Analysis side by side */}
          <div className="sa-charts-row">
            {timeSeries && timeSeries.series.length > 0 && (
              <ErrorBoundary fallbackTitle="Failed to render scatter plot">
                <div className="sa-chart-section sa-chart-half">
                  <h3 className="sa-section-title">Sentiment vs return (per day)</h3>
                  <p className="sa-section-desc">
                    Each point is one trading day; the horizontal axis matches the trailing net sentiment
                    window above (same series as Pearson <em>r</em>). Colour shows quadrant alignment; size
                    reflects mention volume. The dashed line is an ordinary least squares fit through the
                    cloud.
                  </p>
                  <CorrelationScatter
                    data={timeSeries.series}
                    correlationCoefficient={correlation.pearson?.coefficient}
                    height={360}
                    trailingWindowDays={sentimentMemoryDays}
                  />
                </div>
              </ErrorBoundary>
            )}

            {lagAnalysis && lagAnalysis.lags.length > 0 && (
              <ErrorBoundary fallbackTitle="Failed to render lag chart">
                <div className="sa-chart-section sa-chart-half">
                  <h3 className="sa-section-title">Lag analysis</h3>
                  <p className="sa-section-desc">
                    Pearson <em>r</em> between trailing net sentiment and returns at each day offset. Green
                    or red bars are significant at 5%; grey bars are not. The outlined bar is the strongest
                    lag in the grid.
                  </p>
                  <LagChart
                    data={lagAnalysis.lags}
                    bestLag={lagAnalysis.best_lag}
                    height={340}
                  />
                </div>
              </ErrorBoundary>
            )}
          </div>

          {/* Rolling Correlation */}
          {rollingData && rollingData.series && rollingData.series.length > 0 && (
            <ErrorBoundary fallbackTitle="Failed to render rolling correlation">
              <div className="sa-chart-section">
                <h3 className="sa-section-title">Rolling correlation ({rollingData.window}-day window)</h3>
                <p className="sa-section-desc">
                  Time-varying Pearson correlation between trailing net sentiment and returns. Shaded area
                  highlights positive (green) vs negative (red) stretches; dotted lines mark a weak ±0.2
                  band.
                </p>
                <div className="sa-rolling-chart">
                  <RollingCorrelationChart
                    data={rollingData.series}
                    height={340}
                    windowDays={rollingData.window}
                  />
                </div>
                {rollingData.statistics && (
                  <div className="sa-rolling-stats">
                    <span>Mean: {rollingData.statistics.mean_correlation.toFixed(4)}</span>
                    <span>Min: {rollingData.statistics.min_correlation.toFixed(4)}</span>
                    <span>Max: {rollingData.statistics.max_correlation.toFixed(4)}</span>
                    <span>Positive periods: {rollingData.statistics.periods_positive}</span>
                    <span>Negative periods: {rollingData.statistics.periods_negative}</span>
                  </div>
                )}
              </div>
            </ErrorBoundary>
          )}

          {/* Granger Causality */}
          {grangerData && grangerData.summary && !grangerData.error && (
            <ErrorBoundary fallbackTitle="Failed to render Granger analysis">
              <div className="sa-chart-section">
                <h3 className="sa-section-title">Granger Causality Test</h3>
                <p className="sa-section-desc">
                  Tests whether past sentiment helps predict future price movements (and vice versa).
                  This directly addresses whether sentiment is a leading indicator.
                </p>
                <div className="sa-granger-results">
                  <div className="sa-granger-card">
                    <div className="sa-granger-direction">Sentiment → Price</div>
                    <div className={`sa-granger-verdict ${grangerData.summary.sentiment_predicts_price ? 'yes' : 'no'}`}>
                      {grangerData.summary.sentiment_predicts_price ? 'Significant' : 'Not Significant'}
                    </div>
                    {grangerData.summary.best_sentiment_to_price_lag && (
                      <div className="sa-granger-detail">
                        Best lag: {grangerData.summary.best_sentiment_to_price_lag.lag} day(s),
                        F = {grangerData.summary.best_sentiment_to_price_lag.f_statistic.toFixed(2)},
                        p = {grangerData.summary.best_sentiment_to_price_lag.p_value.toFixed(4)}
                      </div>
                    )}
                  </div>
                  <div className="sa-granger-card">
                    <div className="sa-granger-direction">Price → Sentiment</div>
                    <div className={`sa-granger-verdict ${grangerData.summary.price_predicts_sentiment ? 'yes' : 'no'}`}>
                      {grangerData.summary.price_predicts_sentiment ? 'Significant' : 'Not Significant'}
                    </div>
                    {grangerData.summary.best_price_to_sentiment_lag && (
                      <div className="sa-granger-detail">
                        Best lag: {grangerData.summary.best_price_to_sentiment_lag.lag} day(s),
                        F = {grangerData.summary.best_price_to_sentiment_lag.f_statistic.toFixed(2)},
                        p = {grangerData.summary.best_price_to_sentiment_lag.p_value.toFixed(4)}
                      </div>
                    )}
                  </div>
                </div>
                <div className="sa-granger-interpretation">
                  {grangerData.summary.interpretation}
                </div>
              </div>
            </ErrorBoundary>
          )}

          {/* Interpretation */}
          <div className="sa-interpretation">
            <h3 className="sa-section-title">Analysis Interpretation</h3>
            <div className="sa-interpretation-content">
              {renderInterpretation(correlation, lagAnalysis, ticker)}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

function renderInterpretation(
  correlation: CorrelationResponse | null,
  lagAnalysis: LagAnalysisResponse | null,
  ticker: string,
): React.ReactNode {
  if (!correlation) return <p>No analysis data available.</p>;

  const pearson = correlation.pearson;
  const bestLag = lagAnalysis?.best_lag;

  const paragraphs: string[] = [];

  if (pearson) {
    const dir = pearson.coefficient > 0 ? 'positive' : 'negative';
    const strength = Math.abs(pearson.coefficient);

    if (pearson.significant) {
      paragraphs.push(
        `The analysis found a statistically significant ${dir} correlation (r = ${pearson.coefficient.toFixed(4)}, p = ${pearson.p_value.toFixed(4)}) between investor sentiment and ${ticker}'s price movements over the analyzed period.`
      );

      if (strength >= 0.5) {
        paragraphs.push(
          `This is a relatively strong relationship, suggesting that sentiment derived from social media and news sources has meaningful predictive value for ${ticker}'s price direction.`
        );
      } else if (strength >= 0.3) {
        paragraphs.push(
          `This is a moderate relationship. While sentiment shows some predictive value, other factors clearly also influence ${ticker}'s price movements significantly.`
        );
      } else {
        paragraphs.push(
          `However, the correlation is relatively weak, meaning sentiment alone is not a strong predictor of ${ticker}'s price movements. It may be useful as one signal among many.`
        );
      }
    } else {
      paragraphs.push(
        `No statistically significant correlation was found between investor sentiment and ${ticker}'s price movements (r = ${pearson.coefficient.toFixed(4)}, p = ${pearson.p_value.toFixed(4)}). This suggests that for this stock and time period, sentiment data alone does not reliably predict price direction.`
      );
    }
  }

  if (bestLag && bestLag.pearson_r !== null && bestLag.significant) {
    if (bestLag.lag_days > 0) {
      paragraphs.push(
        `The lag analysis suggests sentiment may lead price movements by ${bestLag.lag_days} day(s), with the strongest lagged correlation of r = ${bestLag.pearson_r.toFixed(4)}. This could indicate predictive value of sentiment data for short-term price forecasting.`
      );
    } else if (bestLag.lag_days < 0) {
      paragraphs.push(
        `Interestingly, the lag analysis shows the strongest correlation when price leads sentiment by ${Math.abs(bestLag.lag_days)} day(s) (r = ${bestLag.pearson_r.toFixed(4)}). This suggests price movements may drive subsequent sentiment changes rather than the reverse.`
      );
    } else {
      paragraphs.push(
        `The strongest correlation occurs on the same day, suggesting sentiment and price react to the same events simultaneously rather than one leading the other.`
      );
    }
  }

  if (correlation.data_points < 20) {
    paragraphs.push(
      `Note: This analysis is based on only ${correlation.data_points} overlapping data points. Results may become more reliable with additional data collection over time.`
    );
  }

  return paragraphs.map((p, i) => <p key={i}>{p}</p>);
}

export default StockAnalysis;
