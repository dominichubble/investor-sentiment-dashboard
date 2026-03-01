import React, { useState, useMemo, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Navbar from '../../components/Navbar';
import MetricCard from '../../components/MetricCard';
import SummaryCard from '../../components/SummaryCard';
import { MiniLineChart, SentimentBarChart, MiniAreaChart, CorrelationHeatmap } from '../../components/Charts';
import { ErrorBoundary } from '../../components/ErrorBoundary';
import { DropdownOption } from '../../components/DropdownButton';
import { useDashboard } from '../../context/DashboardContext';
import { apiService, ExplainSentimentResponse } from '../../services/api';
import './Homepage.css';


const assetOptions: DropdownOption[] = [
  { id: 'all', label: 'All Assets (ETFs, Crypto, Stocks)', value: 'all' },
  { id: 'crypto', label: 'Cryptocurrency', value: 'crypto' },
  { id: 'stocks', label: 'Stocks', value: 'stocks' },
  { id: 'etfs', label: 'ETFs', value: 'etfs' }
];

const timeframeOptions: DropdownOption[] = [
  { id: 'all', label: 'All Time', value: 'all' },
  { id: '7d', label: '7 Days', value: '7' },
  { id: '14d', label: '14 Days', value: '14' },
  { id: '30d', label: '30 Days', value: '30' },
  { id: '90d', label: '90 Days', value: '90' },
  { id: '180d', label: '6 Months', value: '180' },
  { id: '365d', label: '1 Year', value: '365' },
];

const Homepage: React.FC = () => {
  const navigate = useNavigate();
  const {
    statistics,
    correlationOverview,
    selectedAssetType,
    selectedTimeframe: ctxTimeframe,
    isLoading,
    error,
    setSelectedAssetType,
    setSelectedTimeframe: setCtxTimeframe,
    refreshData,
  } = useDashboard();

  const [selectedAsset, setSelectedAsset] = useState<DropdownOption>(assetOptions[0]);
  const [selectedTimeframe, setSelectedTimeframe] = useState<DropdownOption>(timeframeOptions[0]);
  const [isExplainOpen, setIsExplainOpen] = useState(false);
  const [explainText, setExplainText] = useState('');
  const [explainResult, setExplainResult] = useState<ExplainSentimentResponse | null>(null);
  const [isExplaining, setIsExplaining] = useState(false);
  const [explainError, setExplainError] = useState<string | null>(null);
  const [explainElapsed, setExplainElapsed] = useState(0);
  const abortRef = useRef<AbortController | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const handleAssetChange = (option: DropdownOption) => {
    setSelectedAsset(option);
    setSelectedAssetType(option.value);
  };

  const handleTimeframeChange = (option: DropdownOption) => {
    setSelectedTimeframe(option);
    setCtxTimeframe(option.value);
  };

  const handleCardClick = (cardType: string) => {
    if (cardType === 'correlation') {
      navigate('/correlation');
    }
  };

  const handleStockClick = (ticker: string) => {
    navigate('/correlation');
  };

  const openExplainModal = () => {
    setIsExplainOpen(true);
    setExplainError(null);
  };

  const closeExplainModal = () => {
    // Cancel any in-flight LIME request when closing the modal
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setIsExplainOpen(false);
    setIsExplaining(false);
    setExplainError(null);
    setExplainElapsed(0);
  };

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (abortRef.current) abortRef.current.abort();
    };
  }, []);

  const handleExplain = async () => {
    if (!explainText.trim()) {
      setExplainError('Please enter text to explain.');
      return;
    }

    // Abort any previous in-flight request
    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setIsExplaining(true);
    setExplainError(null);
    setExplainResult(null);
    setExplainElapsed(0);

    // Start elapsed timer (ticks every second)
    const start = Date.now();
    timerRef.current = setInterval(() => {
      setExplainElapsed(Math.floor((Date.now() - start) / 1000));
    }, 1000);

    try {
      const response = await apiService.explainSentiment(
        explainText,
        { num_features: 12 },
        controller.signal,
      );
      setExplainResult(response);
    } catch (err: any) {
      if (err?.code === 'ERR_CANCELED' || controller.signal.aborted) {
        // User cancelled — no error to show
      } else if (err?.code === 'ECONNABORTED') {
        setExplainError(
          'The explanation timed out. Try shorter text or try again later.'
        );
      } else {
        setExplainError(
          err?.response?.data?.detail || 'Failed to generate explanation. Please try again.'
        );
      }
    } finally {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      setIsExplaining(false);
      abortRef.current = null;
    }
  };

  const formatNumber = (num: number): string => {
    return num.toLocaleString();
  };

  const formatPercentage = (num: number): string => {
    return `${num.toFixed(1)}%`;
  };

  // Derive mini-chart data from real daily_trend (deterministic, memoized)
  const sentimentTrendData = useMemo(() => {
    if (!statistics?.daily_trend || statistics.daily_trend.length === 0) return [];
    return statistics.daily_trend.map(d => d.net_sentiment);
  }, [statistics?.daily_trend]);

  const volumeTrendData = useMemo(() => {
    if (!statistics?.daily_trend || statistics.daily_trend.length === 0) return [];
    return statistics.daily_trend.map(d => d.count);
  }, [statistics?.daily_trend]);

  const calculateNetSentiment = (): string => {
    if (!statistics) return '0.00';
    const dist = statistics.sentiment_distribution;
    const score = ((dist.positive_percentage - dist.negative_percentage) / 100).toFixed(2);
    return Number(score) >= 0 ? `+${score}` : score;
  };

  const getSentimentDescription = (): string => {
    if (!statistics) return 'Awaiting data';
    const score = parseFloat(calculateNetSentiment());
    if (score > 0.5) return 'Strongly Positive';
    if (score > 0.2) return 'Moderately Positive';
    if (score > -0.2) return 'Neutral';
    if (score > -0.5) return 'Moderately Negative';
    return 'Strongly Negative';
  };

  const generateSummaryText = (): string => {
    if (!statistics) return 'Fetching sentiment data from the backend…';
    
    const dist = statistics.sentiment_distribution;
    const score = calculateNetSentiment();
    const topStock = statistics.top_stocks[0];
    const recentActivity = statistics.recent_activity;
    
    return `Over the analyzed period, sentiment across all tracked assets remains ${getSentimentDescription().toLowerCase()} (${score}).

Sentiment is distributed as follows: ${formatPercentage(dist.positive_percentage)} positive, ${formatPercentage(dist.neutral_percentage)} neutral, and ${formatPercentage(dist.negative_percentage)} negative. The most discussed stock is ${topStock?.ticker || 'N/A'} (${topStock?.company_name || 'N/A'}) with ${topStock?.count || 0} mentions.

Recent activity shows ${recentActivity.last_24h} predictions in the last 24 hours, ${recentActivity.last_7d} in the last 7 days, and ${recentActivity.last_30d} in the last 30 days.

The system has analyzed ${formatNumber(statistics.total_predictions)} records across ${statistics.total_stocks_analyzed} unique stocks, providing comprehensive coverage of market sentiment.`;
  };

  // Get the top correlated stock for the summary card
  const getTopCorrelation = () => {
    if (correlationOverview.length === 0) return null;
    return correlationOverview[0];
  };

  if (error && !statistics) {
    return (
      <div className="homepage">
        <Navbar
          assetOptions={assetOptions}
          selectedAsset={selectedAsset}
          onAssetChange={handleAssetChange}
          assetDisabled
          assetDisabledMessage="Coming Soon"
          timeframeOptions={timeframeOptions}
          selectedTimeframe={selectedTimeframe}
          onTimeframeChange={handleTimeframeChange}
        />
        <div className="error-message">
          <h2>Unable to Load Data</h2>
          <p>{error}</p>
          <button onClick={refreshData} className="retry-button">
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (isLoading || !statistics) {
    return (
      <div className="homepage">
        <Navbar
          assetOptions={assetOptions}
          selectedAsset={selectedAsset}
          onAssetChange={handleAssetChange}
          assetDisabled
          assetDisabledMessage="Coming Soon"
          timeframeOptions={timeframeOptions}
          selectedTimeframe={selectedTimeframe}
          onTimeframeChange={handleTimeframeChange}
        />
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
          <p className="loading-text">Loading sentiment data...</p>
        </div>
      </div>
    );
  }

  const netSentiment = calculateNetSentiment();
  const netScore = parseFloat(netSentiment);
  const topCorr = getTopCorrelation();

  return (
    <div className="homepage">
      <Navbar
        assetOptions={assetOptions}
        selectedAsset={selectedAsset}
        onAssetChange={handleAssetChange}
        assetDisabled
        assetDisabledMessage="Coming Soon"
        timeframeOptions={timeframeOptions}
        selectedTimeframe={selectedTimeframe}
        onTimeframeChange={handleTimeframeChange}
      />

      <div className="metrics-grid">
        <MetricCard
          title="NET SENTIMENT SCORE"
          value={netSentiment}
          description={getSentimentDescription()}
          trend={netScore > 0 ? 'up' : netScore < 0 ? 'down' : 'neutral'}
          onClick={() => handleCardClick('net-sentiment')}
          chart={
            <MiniLineChart 
              data={sentimentTrendData} 
              color={netScore > 0 ? "#6cdf7e" : netScore < 0 ? "#cb6e68" : "#8e94a0"}
            />
          }
        />
        
        <MetricCard
          title="SENTIMENT BREAKDOWN"
          value={`${formatPercentage(statistics.sentiment_distribution.positive_percentage)}`}
          description={`Positive leads with ${statistics.sentiment_distribution.positive} records`}
          onClick={() => handleCardClick('sentiment-breakdown')}
          chart={
            <SentimentBarChart
              positive={statistics.sentiment_distribution.positive_percentage}
              neutral={statistics.sentiment_distribution.neutral_percentage}
              negative={statistics.sentiment_distribution.negative_percentage}
            />
          }
        />
        
        <MetricCard
          title="TOTAL RECORDS ANALYSED"
          value={formatNumber(statistics.total_predictions)}
          description={`Tracking ${statistics.total_stocks_analyzed} unique stocks`}
          onClick={() => handleCardClick('total-records')}
          chart={
            <MiniAreaChart 
              data={volumeTrendData} 
              color="#5c7cfa"
            />
          }
        />
        
        <MetricCard
          title="TOP CORRELATION"
          value={topCorr ? `${topCorr.pearson_r >= 0 ? '+' : ''}${topCorr.pearson_r.toFixed(3)}` : 'N/A'}
          description={topCorr
            ? `${topCorr.ticker}: ${topCorr.interpretation}`
            : 'No correlation data yet'}
          trend={topCorr ? (topCorr.pearson_r > 0 ? 'up' : topCorr.pearson_r < 0 ? 'down' : 'neutral') : 'neutral'}
          onClick={() => handleCardClick('correlation')}
        />
      </div>

      <SummaryCard
        title="SENTIMENT ANALYSIS SUMMARY"
        summary={generateSummaryText()}
        embeddedCardTitle="RECENT ACTIVITY"
        embeddedCardValue={formatNumber(statistics.recent_activity.last_7d)}
        embeddedCardDescription={`Predictions in last 7 days (${statistics.recent_activity.last_24h} today)`}
        embeddedCardChart={
          <SentimentBarChart
            positive={statistics.sentiment_distribution.positive_percentage}
            neutral={statistics.sentiment_distribution.neutral_percentage}
            negative={statistics.sentiment_distribution.negative_percentage}
            height={100}
          />
        }
      />

      <div className="explainability-inline">
        <div className="explainability-inline-content">
          <h3>LIME Explainability</h3>
          <p>Generate token-level importance to understand why FinBERT made a sentiment prediction.</p>
        </div>
        <button className="view-all-btn" onClick={openExplainModal}>
          Explain Text
        </button>
      </div>

      {/* Correlation Overview Section */}
      <ErrorBoundary fallbackTitle="Failed to load correlation overview">
        <div className="correlation-section">
          <div className="correlation-section-header">
            <div>
              <h2 className="correlation-section-title">Sentiment-Price Correlations</h2>
              <p className="correlation-section-desc">
                How sentiment aligns with actual stock price movements. Click a stock for detailed analysis.
              </p>
            </div>
            <button className="view-all-btn" onClick={() => navigate('/correlation')}>
              View Full Analysis
            </button>
          </div>
          {correlationOverview.length > 0 ? (
            <CorrelationHeatmap
              data={correlationOverview.slice(0, 12)}
              onStockClick={handleStockClick}
            />
          ) : (
            <div className="correlation-empty">
              <p>No correlation data available for the selected timeframe.</p>
              <p className="correlation-empty-hint">
                Ensure the backend is running and sentiment data has been ingested, then try a broader timeframe.
              </p>
            </div>
          )}
        </div>
      </ErrorBoundary>

      {isExplainOpen && (
        <div className="explain-modal-overlay" onClick={closeExplainModal}>
          <div className="explain-modal" onClick={(e) => e.stopPropagation()}>
            <div className="explain-modal-header">
              <h3>Sentiment Explanation (LIME)</h3>
              <button className="explain-close-btn" onClick={closeExplainModal}>X</button>
            </div>

            <textarea
              className="explain-input"
              placeholder="Paste financial text here..."
              value={explainText}
              onChange={(e) => setExplainText(e.target.value)}
            />

            <div className="explain-actions">
              <button className="retry-button" onClick={handleExplain} disabled={isExplaining}>
                {isExplaining ? 'Explaining...' : 'Run Explanation'}
              </button>
              {!isExplaining && !explainResult && (
                <button
                  className="explain-try-btn"
                  onClick={() => setExplainText('Tesla reported record quarterly revenue and raised its full-year guidance, sending shares up 12% in after-hours trading.')}
                >
                  Try Example
                </button>
              )}
              {isExplaining && (
                <button
                  className="explain-cancel-btn"
                  onClick={() => { if (abortRef.current) abortRef.current.abort(); }}
                >
                  Cancel
                </button>
              )}
            </div>

            {isExplaining && (
              <p className="explain-progress">
                Analysing token importance… {explainElapsed}s elapsed
                {explainElapsed >= 10 && ' — LIME generates hundreds of perturbations; this is normal.'}
              </p>
            )}

            {explainError && <p className="explain-error">{explainError}</p>}

            {explainResult && (
              <div className="explain-result">
                <p className="explain-prediction">
                  Prediction: <strong>{explainResult.prediction.label}</strong> ({(explainResult.prediction.score * 100).toFixed(1)}%)
                  <span className="explain-time"> — {(explainResult.metadata.processing_time_ms / 1000).toFixed(1)}s</span>
                </p>
                <div className="token-list">
                  {explainResult.tokens.map((item) => (
                    <div key={`${item.token}-${item.weight}`} className="token-row">
                      <span className="token">{item.token}</span>
                      <span className={`weight ${item.weight >= 0 ? 'positive' : 'negative'}`}>
                        {item.weight >= 0 ? '+' : ''}{item.weight.toFixed(4)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default Homepage;
