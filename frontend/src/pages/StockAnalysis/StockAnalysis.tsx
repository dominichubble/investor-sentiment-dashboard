import React, { useState, useEffect, useCallback } from 'react';
import * as RechartsPrimitive from 'recharts';
import {
  SentimentPriceChart,
  CorrelationScatter,
  LagChart,
  CorrelationHeatmap,
} from '../../components/Charts';
import { ErrorBoundary } from '../../components/ErrorBoundary';
import Navbar from '../../components/Navbar';
import { apiService } from '../../services/api';
import { useDashboard } from '../../context/DashboardContext';
import type {
  CorrelationResponse,
  LagAnalysisResponse,
  TimeSeriesResponse,
  CorrelationOverviewItem,
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

const StockAnalysis: React.FC = () => {
  const [ticker, setTicker] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const [period, setPeriod] = useState('90d');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

  // Data states
  const [correlation, setCorrelation] = useState<CorrelationResponse | null>(null);
  const [timeSeries, setTimeSeries] = useState<TimeSeriesResponse | null>(null);
  const [lagAnalysis, setLagAnalysis] = useState<LagAnalysisResponse | null>(null);
  const [grangerData, setGrangerData] = useState<GrangerCausalityResponse | null>(null);
  const [rollingData, setRollingData] = useState<RollingCorrelationResponse | null>(null);
  const [stockInfo, setStockInfo] = useState<StockInfoResponse | null>(null);

  // Use shared context for overview data
  const { correlationOverview: overview, isLoading: isOverviewLoading } = useDashboard();

  // UI states
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const analyzeStock = useCallback(async (
    stockTicker: string,
    analysisPeriod: string,
    sd?: string,
    ed?: string,
  ) => {
    if (!stockTicker) return;

    setIsLoading(true);
    setError(null);
    setTicker(stockTicker);

    const isCustom = analysisPeriod === 'custom' && sd && ed;
    const dateParams = isCustom
      ? { start_date: sd, end_date: ed, period: 'max' }
      : { period: analysisPeriod };

    try {
      const [corrData, tsData, lagData, infoData, grangerResult, rollingResult] = await Promise.allSettled([
        apiService.getCorrelation(stockTicker, { ...dateParams }),
        apiService.getCorrelationTimeseries(stockTicker, { ...dateParams }),
        apiService.getLagAnalysis(stockTicker, { max_lag_days: 5, ...dateParams }),
        apiService.getStockInfo(stockTicker),
        apiService.getGrangerCausality(stockTicker, { max_lag: 5, ...dateParams }),
        apiService.getRollingCorrelation(stockTicker, { window: 14, ...dateParams }),
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
  }, []);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchInput.trim()) {
      analyzeStock(searchInput.trim().toUpperCase(), period, startDate, endDate);
    }
  };

  const handleStockClick = (clickedTicker: string) => {
    setSearchInput(clickedTicker);
    analyzeStock(clickedTicker, period, startDate, endDate);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handlePeriodChange = (newPeriod: string) => {
    setPeriod(newPeriod);
    if (newPeriod !== 'custom' && ticker) {
      analyzeStock(ticker, newPeriod);
    }
  };

  const handleCustomDateApply = () => {
    if (ticker && startDate && endDate) {
      analyzeStock(ticker, 'custom', startDate, endDate);
    }
  };

  const getCorrelationColor = (r: number | undefined): string => {
    if (r === undefined) return '#8e94a0';
    if (r > 0.3) return '#7aac86';
    if (r < -0.3) return '#cb6e68';
    return '#8e94a0';
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
      <Navbar title="COC251 Sentiment Analysis" />

      {/* Header / Search */}
      <div className="sa-header">
        <h1 className="sa-title">Sentiment-Price Correlation Analysis</h1>
        <p className="sa-subtitle">
          Analyze the relationship between investor sentiment and stock price movements
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

        <div className="sa-period-selector">
          {PERIOD_OPTIONS.map(opt => (
            <button
              key={opt.value}
              className={`sa-period-btn ${period === opt.value ? 'active' : ''}`}
              onClick={() => handlePeriodChange(opt.value)}
            >
              {opt.label}
            </button>
          ))}
        </div>

        {period === 'custom' && (
          <div className="sa-date-range">
            <div className="sa-date-inputs">
              <label className="sa-date-label">
                From
                <input
                  type="date"
                  className="sa-date-input"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                />
              </label>
              <span className="sa-date-separator">—</span>
              <label className="sa-date-label">
                To
                <input
                  type="date"
                  className="sa-date-input"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                />
              </label>
              <button
                className="sa-date-apply-btn"
                disabled={!startDate || !endDate || !ticker || isLoading}
                onClick={handleCustomDateApply}
              >
                Apply
              </button>
            </div>
          </div>
        )}
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
              <div className="sa-stat-value" style={{ color: '#5c7cfa' }}>
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
                color: correlation.pearson?.significant ? '#7aac86' : '#cb6e68'
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

          {/* Dual-Axis Time Series */}
          {timeSeries && timeSeries.series.length > 0 && (
            <ErrorBoundary fallbackTitle="Failed to render time series chart">
              <div className="sa-chart-section">
                <h3 className="sa-section-title">Sentiment vs Price Over Time</h3>
                <p className="sa-section-desc">
                  Dual-axis view comparing daily net sentiment (right axis) with stock price (left axis).
                  Bars show daily mention volume.
                </p>
                <SentimentPriceChart data={timeSeries.series} height={420} />
              </div>
            </ErrorBoundary>
          )}

          {/* Scatter Plot & Lag Analysis side by side */}
          <div className="sa-charts-row">
            {timeSeries && timeSeries.series.length > 0 && (
              <ErrorBoundary fallbackTitle="Failed to render scatter plot">
                <div className="sa-chart-section sa-chart-half">
                  <h3 className="sa-section-title">Correlation Scatter Plot</h3>
                  <p className="sa-section-desc">
                    Each point is a day: x = net sentiment, y = daily price return.
                    Tighter clustering along a diagonal indicates stronger correlation.
                  </p>
                  <CorrelationScatter
                    data={timeSeries.series}
                    correlationCoefficient={correlation.pearson?.coefficient}
                    height={320}
                  />
                </div>
              </ErrorBoundary>
            )}

            {lagAnalysis && lagAnalysis.lags.length > 0 && (
              <ErrorBoundary fallbackTitle="Failed to render lag chart">
                <div className="sa-chart-section sa-chart-half">
                  <h3 className="sa-section-title">Lag Analysis</h3>
                  <p className="sa-section-desc">
                    Tests if sentiment at day t predicts price at day t+lag.
                    Colored bars are statistically significant (p &lt; 0.05).
                  </p>
                  <LagChart
                    data={lagAnalysis.lags}
                    bestLag={lagAnalysis.best_lag}
                    height={320}
                  />
                </div>
              </ErrorBoundary>
            )}
          </div>

          {/* Rolling Correlation */}
          {rollingData && rollingData.series && rollingData.series.length > 0 && (
            <ErrorBoundary fallbackTitle="Failed to render rolling correlation">
              <div className="sa-chart-section">
                <h3 className="sa-section-title">Rolling Correlation ({rollingData.window}-day window)</h3>
                <p className="sa-section-desc">
                  Shows how the correlation between sentiment and price changes over time.
                  Values above 0 indicate positive correlation; below 0 indicates negative.
                </p>
                <div className="sa-rolling-chart" style={{ height: 320 }}>
                  <RollingCorrelationChart data={rollingData.series} height={320} />
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

      {/* Overview / Heatmap Section */}
      <div className="sa-overview-section">
        <h3 className="sa-section-title">Correlation Overview - All Tracked Stocks</h3>
        <p className="sa-section-desc">
          Click on any stock to analyze its sentiment-price relationship in detail.
          Color intensity reflects correlation strength; grey indicates insufficient data.
        </p>

        {isOverviewLoading ? (
          <div className="sa-loading" style={{ minHeight: 100 }}>
            <div className="sa-spinner" />
            <p>Loading correlation overview...</p>
          </div>
        ) : overview.length > 0 ? (
          <CorrelationHeatmap data={overview} onStockClick={handleStockClick} />
        ) : (
          <div className="sa-empty-overview">
            <p>No stocks with sufficient sentiment data for correlation analysis.</p>
            <p style={{ fontSize: 13, color: '#999' }}>
              Run the data ingestion pipelines and sentiment analysis to populate data.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

function RollingCorrelationChart({ data, height = 300 }: {
  data: { date: string; correlation: number }[];
  height?: number;
}) {
  return (
    <RechartsPrimitive.ResponsiveContainer width="100%" height={height} minWidth={0}>
      <RechartsPrimitive.LineChart data={data} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
        <RechartsPrimitive.CartesianGrid strokeDasharray="3 3" stroke="#e8e8e8" />
        <RechartsPrimitive.XAxis
          dataKey="date"
          tick={{ fontSize: 11, fill: '#888' }}
          tickFormatter={(v: string) => v.slice(5)}
        />
        <RechartsPrimitive.YAxis
          domain={[-1, 1]}
          tick={{ fontSize: 11, fill: '#888' }}
          tickFormatter={(v: number) => v.toFixed(1)}
        />
        <RechartsPrimitive.Tooltip
          contentStyle={{
            background: 'white',
            border: '1px solid #e0e0e0',
            borderRadius: '8px',
            fontSize: '12px',
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
          }}
          formatter={(value: number) => [value.toFixed(4), 'Correlation']}
        />
        <RechartsPrimitive.ReferenceLine y={0} stroke="#ccc" strokeDasharray="3 3" />
        <RechartsPrimitive.Line
          type="monotone"
          dataKey="correlation"
          stroke="#5c7cfa"
          dot={false}
          strokeWidth={2}
        />
      </RechartsPrimitive.LineChart>
    </RechartsPrimitive.ResponsiveContainer>
  );
}

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
