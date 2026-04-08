import React, { useState, useEffect, useCallback, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { useSearchParams } from 'react-router-dom';
import {
  SentimentPriceChart,
  DailySentimentAggregateChart,
  CorrelationScatter,
  LagChart,
  RollingCorrelationChart,
  EmotionTimelineChart,
} from '../../components/Charts';
import { ErrorBoundary } from '../../components/ErrorBoundary';
import Navbar from '../../components/Navbar';
import { apiService, type TickerNarrativeResponse } from '../../services/api';
import {
  scatterYKeyFromEffective,
  type CorrelationResponse,
  type LagAnalysisResponse,
  type TimeSeriesResponse,
  type StockInfoResponse,
  type StockDataQualityResponse,
  type GrangerCausalityResponse,
  type RollingCorrelationResponse,
  type OutOfSampleResponse,
  StockSentimentAggregatedResponse,
} from '../../types';
import './StockAnalysis.css';

/** Preset lookbacks only — custom calendars use the “Choose my own dates” option (no duplicate “Custom” button). */
const PRESET_PERIOD_OPTIONS = [
  { label: '30 days', value: '30d' },
  { label: '90 days', value: '90d' },
  { label: '6 months', value: '6mo' },
  { label: '1 year', value: '1y' },
] as const;

const PRESET_PERIOD_VALUES = new Set(PRESET_PERIOD_OPTIONS.map((o) => o.value));

const EMOTION_COPY: Record<string, string> = {
  fear: 'Negative signals dominate and the text reads risk-off or defensive.',
  optimism: 'The tone leans constructive, with positive expectations or upside language.',
  uncertainty: 'The text is cautious or unresolved, with limited conviction either way.',
  confidence: 'The language suggests stronger conviction in fundamentals or execution.',
  skepticism: 'The stance is doubtful, valuation-aware, or unconvinced by the bullish case.',
  mixed: 'Multiple finance emotions compete, so the dominant read is intentionally cautious.',
};

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

/** Short labels for info-bar tags (full metric name still in API / chart copy). */
function labelForPriceMetric(metric: string | undefined): string {
  switch (metric) {
    case 'forward_1d_return':
      return 'Next-day return';
    case 'forward_excess_return':
      return 'Next-day (ex-SPY)';
    case 'excess_returns':
      return 'Ex-SPY return';
    case 'returns':
      return 'Same-day return';
    default:
      return metric || 'returns';
  }
}

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
  const [dataQuality, setDataQuality] = useState<StockDataQualityResponse | null>(null);
  const [stockSentimentSummary, setStockSentimentSummary] = useState<StockSentimentAggregatedResponse | null>(null);

  // UI states
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  /** Trailing days for net sentiment when comparing to price (causal moving average). */
  const [sentimentMemoryDays, setSentimentMemoryDays] = useState(3);

  /** Correlation methodology (matches backend query params). */
  const [alignMode, setAlignMode] = useState<'same_day' | 'sentiment_leads_1d'>('same_day');
  const [marketAdjustment, setMarketAdjustment] = useState<'none' | 'spy_beta_residual'>('none');
  const [dataSourceFilter, setDataSourceFilter] = useState<string>('');
  const [minMentionsPerDay, setMinMentionsPerDay] = useState(1);

  const [outOfSample, setOutOfSample] = useState<OutOfSampleResponse | null>(null);

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
    const methodology = {
      align_mode: alignMode,
      market_adjustment: marketAdjustment,
      min_mentions_per_day: minMentionsPerDay,
      ...(dataSourceFilter ? { data_source: dataSourceFilter } : {}),
    };
    const rangeParams = customRange
      ? {
          period: analysisPeriod,
          start_date: customRange.start_date,
          end_date: customRange.end_date,
          trailing_days,
          ...methodology,
        }
      : { period: analysisPeriod, trailing_days, ...methodology };

    try {
      const [
        corrData,
        tsData,
        lagData,
        infoData,
        grangerResult,
        rollingResult,
        qualityData,
        oosData,
        emotionSummary,
      ] =
        await Promise.allSettled([
          apiService.getCorrelation(stockTicker, rangeParams),
          apiService.getCorrelationTimeseries(stockTicker, rangeParams),
          apiService.getLagAnalysis(stockTicker, { max_lag_days: 5, ...rangeParams }),
          apiService.getStockInfo(stockTicker),
          apiService.getGrangerCausality(stockTicker, { max_lag: 5, ...rangeParams }),
          apiService.getRollingCorrelation(stockTicker, { window: 14, ...rangeParams }),
          apiService.getStockDataQuality(stockTicker, {
            period: rangeParams.period,
            start_date: 'start_date' in rangeParams ? rangeParams.start_date : undefined,
            end_date: 'end_date' in rangeParams ? rangeParams.end_date : undefined,
            ...(dataSourceFilter ? { data_source: dataSourceFilter } : {}),
          }),
          apiService.getOutOfSampleCorrelation(stockTicker, {
            train_ratio: 0.7,
            ...rangeParams,
          }),
          apiService.getStockSentimentAggregated(stockTicker, {
            period: rangeParams.period,
            start_date: 'start_date' in rangeParams ? rangeParams.start_date : undefined,
            end_date: 'end_date' in rangeParams ? rangeParams.end_date : undefined,
          }),
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

      if (qualityData.status === 'fulfilled') setDataQuality(qualityData.value);
      else setDataQuality(null);

      if (oosData.status === 'fulfilled') setOutOfSample(oosData.value);
      else setOutOfSample(null);

      if (emotionSummary.status === 'fulfilled') setStockSentimentSummary(emotionSummary.value);
      else setStockSentimentSummary(null);

      const allFailed = [corrData, tsData, lagData].every(r => r.status === 'rejected');
      if (allFailed) {
        setError('Could not fetch data for this ticker. Please check the ticker symbol and ensure the backend is running.');
      }

    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred');
    } finally {
      setIsLoading(false);
    }
  }, [
    sentimentMemoryDays,
    alignMode,
    marketAdjustment,
    dataSourceFilter,
    minMentionsPerDay,
  ]);

  useEffect(() => {
    setTickerNarrative(null);
    setTickerNarrativeError(null);
    setDataQuality(null);
    setOutOfSample(null);
    setStockSentimentSummary(null);
  }, [ticker, period, dateRangeMode, customStart, customEnd, alignMode, marketAdjustment, dataSourceFilter, minMentionsPerDay]);

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
      urlPeriod && PRESET_PERIOD_VALUES.has(urlPeriod) ? urlPeriod : '90d';
    if (urlPeriod && PRESET_PERIOD_VALUES.has(urlPeriod)) {
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

  const topEmotion = stockSentimentSummary?.emotion_analysis?.top_emotion ?? 'mixed';
  const topEmotionPct = stockSentimentSummary?.emotion_analysis?.dominant_percentages?.[topEmotion] ?? 0;
  const hasPearson = Boolean(correlation?.pearson);
  const correlationFallback = correlation?.error ?? 'Not enough overlapping days';
  const hasAnyResults = Boolean(
    correlation ||
      (timeSeries && timeSeries.series.length > 0) ||
      (lagAnalysis && lagAnalysis.lags.length > 0) ||
      (rollingData && rollingData.series.length > 0) ||
      (grangerData && grangerData.summary && !grangerData.error) ||
      (outOfSample && !outOfSample.error) ||
      dataQuality ||
      stockSentimentSummary
  );
  const overlappingDays =
    correlation?.data_points ??
    timeSeries?.data_points ??
    rollingData?.data_points ??
    outOfSample?.train?.n ??
    0;
  const effectivePriceMetric =
    correlation?.effective_price_metric ??
    rollingData?.effective_price_metric ??
    outOfSample?.effective_price_metric ??
    'returns';
  const spyBeta =
    correlation?.spy_beta ??
    timeSeries?.spy_beta ??
    outOfSample?.spy_beta ??
    null;
  const effectiveMarketAdjustment =
    correlation?.market_adjustment ??
    timeSeries?.market_adjustment ??
    outOfSample?.market_adjustment ??
    null;

  return (
    <div id="main-content" className="stock-analysis" tabIndex={-1}>
      <Navbar
        title="Sentiment and share price"
        subtitle="Pick how far back to look, or set your own start and end dates. Everything on the page uses the same window. For market-wide mood by source, open Market overview."
      />

      {/* Controls: hero search + time window + analysis options */}
      <div className="sa-header">
        <section className="sa-hero" aria-labelledby="sa-page-title">
          <h1 className="sa-title" id="sa-page-title">
            Analyse a stock
          </h1>
          <p className="sa-subtitle">
            Type a ticker, choose a time window below, then press <strong>Analyse</strong>. If you pick your own dates,
            both days count as part of the range.
          </p>

          <form className="sa-search-form" onSubmit={handleSearch}>
            <input
              type="text"
              className="sa-search-input"
              placeholder="Ticker symbol (e.g. AAPL, TSLA, MSFT)"
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value.toUpperCase())}
              aria-label="Ticker symbol"
            />
            <button type="submit" className="sa-search-btn" disabled={isLoading || !searchInput.trim()}>
              {isLoading ? 'Analysing…' : 'Analyse'}
            </button>
          </form>
        </section>

        <section className="sa-time-window" aria-labelledby="sa-time-window-heading">
          <div className="sa-time-window__head">
            <h2 className="sa-time-window__title" id="sa-time-window-heading">
              Time window
            </h2>
            <p className="sa-time-window__hint">
              Use a preset span or exact calendar dates — charts and stats below all use this same window.
            </p>
          </div>

          <div className="sa-range-mode" role="radiogroup" aria-label="Time window type">
            <label className="sa-range-mode-option">
              <input
                type="radio"
                name="rangeMode"
                checked={dateRangeMode === 'preset'}
                onChange={() => setDateRangeMode('preset')}
              />
              <span>Fixed lookback</span>
            </label>
            <label className="sa-range-mode-option">
              <input
                type="radio"
                name="rangeMode"
                checked={dateRangeMode === 'custom'}
                onChange={() => setDateRangeMode('custom')}
              />
              <span>Choose my own dates</span>
            </label>
          </div>

          {dateRangeMode === 'preset' && (
            <div className="sa-period-selector" role="group" aria-label="Lookback length">
              {PRESET_PERIOD_OPTIONS.map((opt) => (
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
                <span className="sa-date-label">Start</span>
                <input
                  type="date"
                  className="sa-date-input"
                  value={customStart}
                  max={customEnd || todayISO}
                  onChange={(e) => setCustomStart(e.target.value)}
                />
              </label>
              <label className="sa-date-field">
                <span className="sa-date-label">End</span>
                <input
                  type="date"
                  className="sa-date-input"
                  value={customEnd}
                  min={customStart}
                  max={todayISO}
                  onChange={(e) => setCustomEnd(e.target.value)}
                />
              </label>
              <p className="sa-custom-dates__hint" id="sa-custom-dates-hint">
                After you change dates, press <strong>Analyse</strong> again to refresh charts and stats (one place — no
                separate “apply dates” button).
              </p>
            </div>
          )}
        </section>

        <section className="sa-analysis-settings" aria-labelledby="sa-settings-title">
          <header className="sa-analysis-settings__intro">
            <p className="sa-analysis-settings__kicker">After you run Analyse</p>
            <h2 className="sa-analysis-settings__title" id="sa-settings-title">
              Fine-tune mood and how it meets price
            </h2>
            <p className="sa-analysis-settings__lede">
              <strong>Smoothing</strong> blends nearby calendar days into each day’s mood score and refreshes charts and
              stats immediately. <strong>Compare and filter</strong> changes timing vs price, market backdrop, channel,
              and how many mentions count — use <strong>Update analysis</strong> to apply those.
            </p>
          </header>

          <div className="sa-settings-smooth" role="region" aria-labelledby="sa-memory-title">
            <div className="sa-settings-smooth__copy">
              <h3 className="sa-settings-block-title" id="sa-memory-title">
                Rolling mood window
              </h3>
              <p className="sa-settings-block-desc">
                More days = a calmer line; one day = raw daily scores. Future dates are never used.
              </p>
            </div>
            <div className="sa-settings-smooth__control">
              <span className="sa-settings-control-label" id="sa-memory-segment-label">
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

          <hr className="sa-analysis-settings__rule" aria-hidden="true" />

          <div className="sa-settings-refine" role="region" aria-labelledby="sa-method-title">
            <div className="sa-settings-refine__head">
              <h3 className="sa-settings-block-title" id="sa-method-title">
                Compare to price and filter posts
              </h3>
              <p className="sa-settings-block-desc">
                Same day vs next trading day, optional SPY strip-out, which channel to use, and a minimum mentions
                threshold for thin days.
              </p>
            </div>
            <div className="sa-methodology-grid">
            <label className="sa-method-field">
              <span className="sa-method-label">Timing vs price</span>
              <select
                className="sa-method-select"
                value={alignMode}
                onChange={(e) => setAlignMode(e.target.value as typeof alignMode)}
              >
                <option value="same_day">Same trading day</option>
                <option value="sentiment_leads_1d">Mood today → next trading day’s move</option>
              </select>
            </label>
            <label className="sa-method-field">
              <span className="sa-method-label">Market backdrop</span>
              <select
                className="sa-method-select"
                value={marketAdjustment}
                onChange={(e) => setMarketAdjustment(e.target.value as typeof marketAdjustment)}
              >
                <option value="none">Stock only (no market strip-out)</option>
                <option value="spy_beta_residual">Remove typical SPY co-movement</option>
              </select>
            </label>
            <label className="sa-method-field">
              <span className="sa-method-label">Sentiment channel</span>
              <select
                className="sa-method-select"
                value={dataSourceFilter}
                onChange={(e) => setDataSourceFilter(e.target.value)}
              >
                <option value="">All channels</option>
                <option value="reddit">reddit</option>
                <option value="news">news</option>
                <option value="twitter">twitter</option>
              </select>
            </label>
            <label className="sa-method-field">
              <span className="sa-method-label">Minimum mentions per day</span>
              <input
                type="number"
                className="sa-method-input"
                min={1}
                max={500}
                value={minMentionsPerDay}
                onChange={(e) => setMinMentionsPerDay(Math.max(1, Math.min(500, Number(e.target.value) || 1)))}
              />
            </label>
          </div>
          <div className="sa-methodology-actions">
            <button
              type="button"
              className="sa-apply-method-btn"
              disabled={!ticker || isLoading}
              onClick={() => {
                if (!ticker) return;
                const custom: CustomRangePayload =
                  dateRangeMode === 'custom' && customStart && customEnd
                    ? { start_date: customStart, end_date: customEnd }
                    : null;
                analyzeStock(ticker, period, custom);
              }}
            >
              Update analysis
            </button>
            <div className="sa-method-presets">
              <button
                type="button"
                className="sa-period-btn"
                onClick={() => {
                  setAlignMode('same_day');
                  setMarketAdjustment('none');
                }}
              >
                Quick reset: simple
              </button>
              <button
                type="button"
                className="sa-period-btn"
                onClick={() => {
                  setAlignMode('sentiment_leads_1d');
                  setMarketAdjustment('spy_beta_residual');
                }}
              >
                Quick reset: next-day + market
              </button>
            </div>
          </div>
          </div>
        </section>
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
          <p>Loading prices and lining up sentiment for {ticker}…</p>
        </div>
      )}

      {!isLoading && ticker && !hasAnyResults && error && (
        <div className="sa-results sa-results-partial">
          <p className="sa-partial-message">{error}</p>
        </div>
      )}

      {/* Results */}
      {!isLoading && ticker && hasAnyResults && (
        <div className="sa-results">
          <header className="sa-results__header">
            <p className="sa-results__kicker">Results</p>
            <h2 className="sa-results__title visually-hidden">Analysis output for {ticker}</h2>
          </header>
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
              {overlappingDays > 0 && (
                <span className="sa-tag">{overlappingDays} overlapping days</span>
              )}
              {effectivePriceMetric && (
                <span
                  className="sa-tag"
                  title={`Technical id: ${effectivePriceMetric}`}
                >
                  Price view: {labelForPriceMetric(effectivePriceMetric)}
                </span>
              )}
              {spyBeta != null && effectiveMarketAdjustment === 'spy_beta_residual' && (
                <span
                  className="sa-tag"
                  title="How much this stock typically moves with SPY in this window; used to peel out broad market wiggle."
                >
                  Versus SPY (typical co-movement) ~ {Number(spyBeta).toFixed(3)}
                </span>
              )}
            </div>
          </div>

          {dataQuality && (
            <div className="sa-chart-section sa-data-quality" aria-label="Sentiment data quality">
              <h3 className="sa-section-title">How trustworthy is the text feed?</h3>
              <p className="sa-section-desc">
                Built from the same mentions as your charts and AI summary, over the <strong>same dates</strong>. Flags
                are simple rules (how much text, which channels, gaps in the calendar) — not formal statistical tests.
              </p>
              {dataQuality.error ? (
                <p className="sa-data-quality__error" role="alert">
                  {dataQuality.error}
                </p>
              ) : (
                <>
                  <div className="sa-data-quality__top">
                    <div className="sa-data-quality__score-block">
                      <div className="sa-data-quality__score-label">Reliability (rule-of-thumb)</div>
                      <div
                        className={`sa-data-quality__badge sa-data-quality__badge--${dataQuality.confidence_label}`}
                      >
                        {(dataQuality.confidence_label || '—').replace(/^\w/, (c) => c.toUpperCase())}
                      </div>
                      <div className="sa-data-quality__meter-wrap" aria-hidden>
                        <div
                          className="sa-data-quality__meter-fill"
                          style={{
                            width: `${Math.min(100, Math.round(dataQuality.confidence_score * 100))}%`,
                          }}
                        />
                      </div>
                      <div className="sa-data-quality__score-num">
                        Score {(dataQuality.confidence_score * 100).toFixed(0)}% — use alongside charts and fundamentals,
                        not on its own.
                      </div>
                    </div>
                    <dl className="sa-data-quality__stats">
                      <div>
                        <dt>Mentions in window</dt>
                        <dd>{dataQuality.total_mentions.toLocaleString()}</dd>
                      </div>
                      <div>
                        <dt>Days with ≥1 mention</dt>
                        <dd>
                          {dataQuality.days_with_mentions} / {dataQuality.calendar_days}
                        </dd>
                      </div>
                      <div>
                        <dt>Calendar coverage</dt>
                        <dd>{(dataQuality.calendar_coverage * 100).toFixed(0)}%</dd>
                      </div>
                      <div>
                        <dt>Longest gap (no mentions)</dt>
                        <dd>{dataQuality.longest_gap_days} day(s)</dd>
                      </div>
                    </dl>
                  </div>
                  <div className="sa-data-quality__row">
                    <div className="sa-data-quality__col">
                      <h4 className="sa-data-quality__subhead">Model label mix (FinBERT)</h4>
                      <ul className="sa-data-quality__mini">
                        <li>
                          Positive ~{(100 * (dataQuality.label_shares?.positive ?? 0)).toFixed(0)}% (
                          {dataQuality.by_label?.positive ?? 0})
                        </li>
                        <li>
                          Neutral ~{(100 * (dataQuality.label_shares?.neutral ?? 0)).toFixed(0)}% (
                          {dataQuality.by_label?.neutral ?? 0})
                        </li>
                        <li>
                          Negative ~{(100 * (dataQuality.label_shares?.negative ?? 0)).toFixed(0)}% (
                          {dataQuality.by_label?.negative ?? 0})
                        </li>
                      </ul>
                    </div>
                    <div className="sa-data-quality__col">
                      <h4 className="sa-data-quality__subhead">Share by source channel</h4>
                      {dataQuality.total_mentions > 0 &&
                      Object.keys(dataQuality.by_channel).length > 0 ? (
                        <div className="sa-data-quality__channels">
                          {Object.entries(dataQuality.by_channel)
                            .sort((a, b) => b[1] - a[1])
                            .map(([ch, n]) => {
                              const pct = Math.round((n / dataQuality.total_mentions) * 100);
                              const safe = ch.replace(/[^a-z]/g, '') || 'unknown';
                              return (
                                <div key={ch} className="sa-data-quality__channel-row">
                                  <span className="sa-data-quality__ch-name">{ch}</span>
                                  <div className="sa-data-quality__channel-bar-wrap">
                                    <div
                                      className={`sa-data-quality__channel-bar sa-data-quality__channel-bar--${safe}`}
                                      style={{ width: `${pct}%` }}
                                    />
                                  </div>
                                  <span className="sa-data-quality__ch-pct">{pct}%</span>
                                </div>
                              );
                            })}
                        </div>
                      ) : (
                        <p className="sa-data-quality__empty-ch">No channel breakdown (no rows).</p>
                      )}
                    </div>
                  </div>
                  {dataQuality.flags.length > 0 && (
                    <ul className="sa-data-quality__flags">
                      {dataQuality.flags.map((f) => (
                        <li
                          key={`${f.id}-${f.title}`}
                          className={`sa-data-quality__flag sa-data-quality__flag--${f.severity}`}
                        >
                          <strong>{f.title}</strong>
                          <span>{f.detail}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </>
              )}
            </div>
          )}

          {stockSentimentSummary?.emotion_analysis && (
            <div className="sa-chart-section" aria-label="Finance emotion summary">
              <h3 className="sa-section-title">Finance emotion layer</h3>
              <p className="sa-section-desc">
                A hybrid finance-emotion classifier built from FinBERT sentiment, uncertainty, aspect evidence,
                and market-language lexicon scoring. The page shows one dominant emotion per mention in the same
                window as the rest of this analysis.
              </p>
              <div className="sa-summary-grid">
                <div className="sa-stat-card">
                  <div className="sa-stat-label">Dominant emotion</div>
                  <div className="sa-stat-value" style={{ color: 'var(--color-accent)' }}>
                    {topEmotion.replace(/^\w/, (c) => c.toUpperCase())}
                  </div>
                  <div className="sa-stat-detail">{EMOTION_COPY[topEmotion] ?? EMOTION_COPY.mixed}</div>
                  <div className="sa-stat-badge significant">
                    {topEmotionPct.toFixed(1)}% of mentions
                  </div>
                </div>
                <div className="sa-stat-card">
                  <div className="sa-stat-label">Fear</div>
                  <div className="sa-stat-value" style={{ color: 'var(--color-negative)' }}>
                    {stockSentimentSummary.emotion_analysis.dominant_distribution.fear ?? 0}
                  </div>
                  <div className="sa-stat-detail">
                    {stockSentimentSummary.emotion_analysis.dominant_percentages.fear?.toFixed(1) ?? '0.0'}% of mentions
                  </div>
                </div>
                <div className="sa-stat-card">
                  <div className="sa-stat-label">Optimism</div>
                  <div className="sa-stat-value" style={{ color: 'var(--color-positive)' }}>
                    {stockSentimentSummary.emotion_analysis.dominant_distribution.optimism ?? 0}
                  </div>
                  <div className="sa-stat-detail">
                    {stockSentimentSummary.emotion_analysis.dominant_percentages.optimism?.toFixed(1) ?? '0.0'}% of mentions
                  </div>
                </div>
                <div className="sa-stat-card">
                  <div className="sa-stat-label">Mixed / uncertain</div>
                  <div className="sa-stat-value" style={{ color: 'var(--color-text-muted)' }}>
                    {(stockSentimentSummary.emotion_analysis.dominant_distribution.mixed ?? 0) +
                      (stockSentimentSummary.emotion_analysis.dominant_distribution.uncertainty ?? 0)}
                  </div>
                  <div className="sa-stat-detail">
                    Combined unresolved or ambiguous emotion reads in this window
                  </div>
                </div>
              </div>

              {stockSentimentSummary.emotion_analysis.timeline.length > 0 && (
                <div className="sa-chart-section" style={{ marginTop: 24, marginBottom: 0 }}>
                  <h4 className="sa-subsection-title">Emotion timeline</h4>
                  <p className="sa-section-desc sa-section-desc--tight">
                    Stacked bars show the count of dominant emotions per day, while the line tracks total
                    mentions. This is useful for spotting shifts from optimism to skepticism, or spikes in fear
                    around volatile periods.
                  </p>
                  <EmotionTimelineChart
                    data={stockSentimentSummary.emotion_analysis.timeline}
                    height={340}
                  />
                </div>
              )}
            </div>
          )}

          {/* Correlation Summary Cards */}
          {correlation ? (
          <div className="sa-summary-grid">
            <div className="sa-stat-card">
              <div
                className="sa-stat-label"
                title="Pearson r: how tightly sentiment and returns line up on a straight trend, from −1 (move opposite) to +1 (move together)."
              >
                Line fit (Pearson <em>r</em>)
              </div>
              <div
                className="sa-stat-value"
                style={{ color: getCorrelationColor(correlation.pearson?.coefficient) }}
              >
                {correlation.pearson
                  ? `${correlation.pearson.coefficient >= 0 ? '+' : ''}${correlation.pearson.coefficient.toFixed(4)}`
                  : 'N/A'}
              </div>
              <div className="sa-stat-detail">
                {correlation.pearson?.interpretation || correlationFallback}
              </div>
              {correlation.pearson && (
                <div
                  className={`sa-stat-badge ${correlation.pearson.significant ? 'significant' : 'not-significant'}`}
                  title="p-value: rough chance you would see a line this strong if nothing were really going on. Small p → less likely to be a fluke (common cut-off 5%)."
                >
                  {correlation.pearson.significant
                    ? `p = ${correlation.pearson.p_value.toFixed(4)}`
                    : 'No strong evidence'}
                </div>
              )}
            </div>

            <div className="sa-stat-card">
              <div
                className="sa-stat-label"
                title="Spearman: same story as Pearson, but uses ranked order — a few wild outlier days pull less weight."
              >
                Rank-based link (Spearman)
              </div>
              <div
                className="sa-stat-value"
                style={{ color: getCorrelationColor(correlation.spearman?.coefficient) }}
              >
                {correlation.spearman
                  ? `${correlation.spearman.coefficient >= 0 ? '+' : ''}${correlation.spearman.coefficient.toFixed(4)}`
                  : 'N/A'}
              </div>
              <div className="sa-stat-detail">
                {correlation.spearman?.interpretation || correlationFallback}
              </div>
              {correlation.spearman && (
                <div
                  className={`sa-stat-badge ${correlation.spearman.significant ? 'significant' : 'not-significant'}`}
                  title="p-value for the rank-based link; small p suggests the pattern is unlikely to be pure chance."
                >
                  {correlation.spearman.significant
                    ? `p = ${correlation.spearman.p_value.toFixed(4)}`
                    : 'No strong evidence'}
                </div>
              )}
            </div>

            <div className="sa-stat-card">
              <div
                className="sa-stat-label"
                title="Shifts sentiment forward or backward by whole trading days to see when the clearest link appears."
              >
                Strongest timing offset
              </div>
              <div className="sa-stat-value" style={{ color: 'var(--color-accent)' }}>
                {lagAnalysis?.best_lag
                  ? lagAnalysis.best_lag.lag_days === 0
                    ? 'Same day'
                    : lagAnalysis.best_lag.lag_days > 0
                      ? `Mood leads by ${lagAnalysis.best_lag.lag_days}d`
                      : `Mood lags by ${Math.abs(lagAnalysis.best_lag.lag_days)}d`
                  : 'N/A'}
              </div>
              <div className="sa-stat-detail">
                {lagAnalysis?.best_lag
                  ? `At that offset, r ≈ ${lagAnalysis.best_lag.pearson_r?.toFixed(4)}`
                  : 'No timing grid for this window'}
              </div>
              {lagAnalysis?.best_lag?.significant && (
                <div className="sa-stat-badge significant">
                  Sentiment {lagAnalysis.best_lag.lag_days > 0 ? 'leads' : 'follows'} price
                </div>
              )}
            </div>

            <div className="sa-stat-card">
              <div
                className="sa-stat-label"
                title="Plain read on whether the Pearson line is strong enough to take seriously in a textbook stats sense."
              >
                Plain-English readout
              </div>
              <div className="sa-stat-value" style={{
                color: !hasPearson
                  ? 'var(--color-text-muted)'
                  : correlation.pearson?.significant
                    ? 'var(--color-positive)'
                    : 'var(--color-negative)'
              }}>
                {!hasPearson ? 'Needs more data' : correlation.pearson?.significant ? 'Looks real' : 'Inconclusive'}
              </div>
              <div className="sa-stat-detail">
                {!hasPearson
                  ? correlationFallback
                  : correlation.pearson?.significant
                  ? 'The straight-line pattern is unlikely to be pure chance (usual 5% bar).'
                  : 'We cannot rule out chance for the straight-line pattern in this slice.'}
              </div>
            </div>
          </div>
          ) : (
            <div className="sa-chart-section">
              <h3 className="sa-section-title">Correlation summary</h3>
              <p className="sa-section-desc">
                Correlation metrics are unavailable for this request, but any charts below still reflect the data that could be loaded.
              </p>
              {error && <p className="sa-data-quality__error" role="alert">{error}</p>}
            </div>
          )}

          <div className="sa-chart-section sa-ticker-narrative">
            <h3 className="sa-section-title">AI plain-language summary</h3>
            <p className="sa-section-desc">
              Reads <strong>only the posts and articles we already stored</strong> for the same dates as your charts
              (Groq API). Results are cached per ticker until the underlying data changes; add{' '}
              <code className="sa-inline-code">GROQ_API_KEY</code> to the backend <code className="sa-inline-code">.env</code>{' '}
              to enable it.
            </p>
            <div className="sa-ticker-narrative__actions">
              <button
                type="button"
                className="sa-narrative-btn"
                disabled={tickerNarrativeLoading}
                onClick={() => fetchTickerNarrative(false)}
              >
                {tickerNarrativeLoading ? 'Working…' : 'Fetch summary (reuse cache if nothing changed)'}
              </button>
              <button
                type="button"
                className="sa-narrative-btn sa-narrative-btn--secondary"
                disabled={tickerNarrativeLoading}
                onClick={() => fetchTickerNarrative(true)}
              >
                Regenerate (fresh API call)
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
                <h3 className="sa-section-title">Mood by calendar day</h3>
                <p className="sa-section-desc">
                  Each day, every mention of {ticker} is pooled. The coloured bands show the split between upbeat,
                  neutral, and downbeat labels; the line is net mood (upbeat share minus downbeat share), on a −1 to
                  +1 scale.
                </p>
                <DailySentimentAggregateChart data={timeSeries.series} height={400} />
              </div>
            </ErrorBoundary>
          )}

          {/* Dual-Axis Time Series — trailing sentiment vs price */}
          {timeSeries && timeSeries.series.length > 0 && (
            <ErrorBoundary fallbackTitle="Failed to render time series chart">
              <div className="sa-chart-section">
                <h3 className="sa-section-title">Smoothed mood vs closing price</h3>
                <p className="sa-section-desc">
                  The <strong>solid</strong> trace is a <strong>rolling average</strong> of daily net mood: each point
                  blends that day with earlier days in range (never future ones). A stretch of upbeat chat lifts the
                  line before you compare it to the <strong>closing</strong> price on the left axis. If you pick more
                  than one day of smoothing, the <strong>dashed</strong> line shows same-day net only. Bars are how
                  many mentions arrived each day. Tweak the smoothing control above to refit everything.
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
                  <h3 className="sa-section-title">One dot per trading day</h3>
                  <p className="sa-section-desc">
                    Across: smoothed net mood (same window as the headline line fit). Up: that day’s return using the
                    same price definition as the summary (
                    <strong>{effectivePriceMetric}</strong>
                    ). Dot colour shows whether mood and return pointed the same way; size scales with how much was
                    posted. The dashed line is a simple best straight-line fit through the cloud (ordinary least
                    squares).
                  </p>
                  <CorrelationScatter
                    data={timeSeries.series}
                    correlationCoefficient={correlation?.pearson?.coefficient}
                    height={360}
                    trailingWindowDays={sentimentMemoryDays}
                    yReturnKey={scatterYKeyFromEffective(effectivePriceMetric)}
                  />
                </div>
              </ErrorBoundary>
            )}

            {lagAnalysis && lagAnalysis.lags.length > 0 && (
              <ErrorBoundary fallbackTitle="Failed to render lag chart">
                <div className="sa-chart-section sa-chart-half">
                  <h3 className="sa-section-title">Timing grid</h3>
                  <p className="sa-section-desc">
                    For each whole-day shift, we measure how tightly mood and returns line up (Pearson <em>r</em>).
                    Green or red bars pass a usual 5% “could this be chance?” check; grey bars do not. The outlined
                    bar is the strongest offset in this window.
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
                <h3 className="sa-section-title">How the link drifts over time ({rollingData.window}-day window)</h3>
                <p className="sa-section-desc">
                  Each point is the straight-line link strength (Pearson) between smoothed mood and{' '}
                  <strong>{rollingData.effective_price_metric ?? 'returns'}</strong> using only the surrounding{' '}
                  {rollingData.window} sessions. Green stretches tilt positive; red stretches tilt negative; faint
                  lines mark a weak ±0.2 band so you can eyeball “mostly noise” zones.
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
                    <span>Average: {rollingData.statistics.mean_correlation.toFixed(4)}</span>
                    <span>Lowest: {rollingData.statistics.min_correlation.toFixed(4)}</span>
                    <span>Highest: {rollingData.statistics.max_correlation.toFixed(4)}</span>
                    <span>Intervals &gt; 0: {rollingData.statistics.periods_positive}</span>
                    <span>Intervals &lt; 0: {rollingData.statistics.periods_negative}</span>
                  </div>
                )}
              </div>
            </ErrorBoundary>
          )}

          {/* Granger Causality */}
          {grangerData && grangerData.summary && !grangerData.error && (
            <ErrorBoundary fallbackTitle="Failed to render Granger analysis">
              <div className="sa-chart-section">
                <h3 className="sa-section-title">Does past mood help forecast price?</h3>
                <p className="sa-section-desc">
                  Granger-style check (stats jargon): asks whether yesterday’s mood series carries extra information
                  for tomorrow’s price move, and the reverse. Handy for “leading indicator?” questions — not a trading
                  signal on its own.
                </p>
                <div className="sa-granger-results">
                  <div className="sa-granger-card">
                    <div className="sa-granger-direction">Mood → price</div>
                    <div className={`sa-granger-verdict ${grangerData.summary.sentiment_predicts_price ? 'yes' : 'no'}`}>
                      {grangerData.summary.sentiment_predicts_price ? 'Adds information' : 'No extra edge'}
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
                    <div className="sa-granger-direction">Price → mood</div>
                    <div className={`sa-granger-verdict ${grangerData.summary.price_predicts_sentiment ? 'yes' : 'no'}`}>
                      {grangerData.summary.price_predicts_sentiment ? 'Adds information' : 'No extra edge'}
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

          {outOfSample && !outOfSample.error && outOfSample.train && outOfSample.test && (
            <div className="sa-chart-section sa-oos-section">
              <h3 className="sa-section-title">Sanity check: older vs newer days</h3>
              <p className="sa-section-desc">
                Days are sorted in time; the first <strong>{Math.round((outOfSample.train_ratio ?? 0.7) * 100)}%</strong>{' '}
                form the <strong>older</strong> bucket and the rest are <strong>held back</strong> (split after{' '}
                <strong>{outOfSample.split_date}</strong>). Comparing Pearson <em>r</em> on the held-back slice shows
                whether the straight-line link still shows up more recently — it is <strong>not</strong> a trading
                backtest.
              </p>
              <div className="sa-oos-grid">
                <div className="sa-oos-card">
                  <div className="sa-oos-card__label">Older slice</div>
                  <div className="sa-oos-card__main">
                    <em>r</em> = {outOfSample.train.pearson_r?.toFixed(4) ?? '—'},{' '}
                    <em>p</em> = {outOfSample.train.pearson_p?.toFixed(4) ?? '—'}
                  </div>
                  <div className="sa-oos-card__sub">{outOfSample.train.n} days</div>
                </div>
                <div className="sa-oos-card">
                  <div className="sa-oos-card__label">Held-back slice</div>
                  <div className="sa-oos-card__main">
                    <em>r</em> = {outOfSample.test.pearson_r?.toFixed(4) ?? '—'},{' '}
                    <em>p</em> = {outOfSample.test.pearson_p?.toFixed(4) ?? '—'}
                  </div>
                  <div className="sa-oos-card__sub">{outOfSample.test.n} days</div>
                </div>
              </div>
            </div>
          )}

          {/* Interpretation */}
          <div className="sa-interpretation">
            <h3 className="sa-section-title">Plain-English wrap-up</h3>
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
  if (!correlation) return <p>No results loaded yet — run an analysis above.</p>;

  if (!correlation.pearson || !correlation.spearman) {
    return (
      <p>
        {correlation.error ||
          `There is not enough overlapping sentiment and price data to estimate a stable correlation for ${ticker} in this window yet.`}
      </p>
    );
  }

  const pearson = correlation.pearson;
  const bestLag = lagAnalysis?.best_lag;

  const paragraphs: string[] = [];

  if (pearson) {
    const dir = pearson.coefficient > 0 ? 'positive' : 'negative';
    const strength = Math.abs(pearson.coefficient);

    if (pearson.significant) {
      paragraphs.push(
        `Over this window, chat and news mood moved in a ${dir} straight-line pattern with ${ticker}’s price (r = ${pearson.coefficient.toFixed(4)}; p = ${pearson.p_value.toFixed(4)} — smaller p usually means “less likely to be a fluke”).`
      );

      if (strength >= 0.5) {
        paragraphs.push(
          `That link is fairly tight for a messy real-world feed, so mood may deserve a seat at the table — still alongside fundamentals, liquidity, and everything else that drives the share.`
        );
      } else if (strength >= 0.3) {
        paragraphs.push(
          `The link is middling: the feed carries some signal, but plenty of other forces still dominate how ${ticker} trades day to day.`
        );
      } else {
        paragraphs.push(
          `Even though the pattern clears the usual “chance” bar, the line is still shallow — treat the feed as one soft input among many, not a crystal ball.`
        );
      }
    } else {
      paragraphs.push(
        `We did not find a straight-line relationship strong enough to dismiss luck (r = ${pearson.coefficient.toFixed(4)}, p = ${pearson.p_value.toFixed(4)}). For this ticker and slice of history, mood in the scraped text alone is not a dependable guide to direction.`
      );
    }
  }

  if (bestLag && bestLag.pearson_r !== null && bestLag.significant) {
    if (bestLag.lag_days > 0) {
      paragraphs.push(
        `Shifting mood forward by ${bestLag.lag_days} trading day(s) lines up best with returns (r ≈ ${bestLag.pearson_r.toFixed(4)}), which hints that posts might lead the next session — worth a cautious read, not a trade rule.`
      );
    } else if (bestLag.lag_days < 0) {
      paragraphs.push(
        `The cleanest fit appears when price leads mood by ${Math.abs(bestLag.lag_days)} day(s) (r ≈ ${bestLag.pearson_r.toFixed(4)}), which suggests traders talk after the tape moves, not the other way around.`
      );
    } else {
      paragraphs.push(
        `The same-day pairing is strongest, which often means sentiment and price are reacting together to fresh news rather than one clearly dragging the other.`
      );
    }
  }

  if (correlation.data_points < 20) {
    paragraphs.push(
      `Heads-up: only ${correlation.data_points} overlapping days fed this view — widen the window or wait for more chatter before leaning on it heavily.`
    );
  }

  return paragraphs.map((p, i) => <p key={i}>{p}</p>);
}

export default StockAnalysis;
