import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Navbar from '../../components/Navbar';
import MetricCard from '../../components/MetricCard';
import SummaryCard from '../../components/SummaryCard';
import { MiniLineChart, SentimentBarChart, MiniAreaChart, CorrelationHeatmap } from '../../components/Charts';
import { ErrorBoundary } from '../../components/ErrorBoundary';
import { DropdownOption } from '../../components/DropdownButton';
import { useDashboard } from '../../context/DashboardContext';
import './Homepage.css';


const assetOptions: DropdownOption[] = [
  { id: 'all', label: 'All Assets (ETFs, Crypto, Stocks)', value: 'all' },
  { id: 'crypto', label: 'Cryptocurrency', value: 'crypto' },
  { id: 'stocks', label: 'Stocks', value: 'stocks' },
  { id: 'etfs', label: 'ETFs', value: 'etfs' }
];

const timeframeOptions: DropdownOption[] = [
  { id: '7d', label: '7 Days', value: '7' },
  { id: '14d', label: '14 Days', value: '14' },
  { id: '30d', label: '30 Days', value: '30' },
  { id: '90d', label: '90 Days', value: '90' }
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

  const formatNumber = (num: number): string => {
    return num.toLocaleString();
  };

  const formatPercentage = (num: number): string => {
    return `${num.toFixed(1)}%`;
  };

  const generateTrendData = (days: number, trend: 'up' | 'down' | 'stable'): number[] => {
    const data: number[] = [];
    let value = 0.2 + Math.random() * 0.3;
    
    for (let i = 0; i < days; i++) {
      if (trend === 'up') {
        value += (Math.random() * 0.1 - 0.03);
      } else if (trend === 'down') {
        value += (Math.random() * 0.1 - 0.07);
      } else {
        value += (Math.random() * 0.1 - 0.05);
      }
      data.push(Math.max(0.1, Math.min(0.9, value)));
    }
    return data;
  };

  const calculateNetSentiment = (): string => {
    if (!statistics) return '0.00';
    const dist = statistics.sentiment_distribution;
    const score = ((dist.positive_percentage - dist.negative_percentage) / 100).toFixed(2);
    return Number(score) >= 0 ? `+${score}` : score;
  };

  const getSentimentDescription = (): string => {
    if (!statistics) return 'Loading...';
    const score = parseFloat(calculateNetSentiment());
    if (score > 0.5) return 'Strongly Positive';
    if (score > 0.2) return 'Moderately Positive';
    if (score > -0.2) return 'Neutral';
    if (score > -0.5) return 'Moderately Negative';
    return 'Strongly Negative';
  };

  const generateSummaryText = (): string => {
    if (!statistics) return 'Loading data...';
    
    const dist = statistics.sentiment_distribution;
    const score = calculateNetSentiment();
    const topStock = statistics.top_stocks[0];
    const recentActivity = statistics.recent_activity;
    
    return `Over the analyzed period, sentiment across all tracked assets remains ${getSentimentDescription().toLowerCase()} (${score}).

Sentiment is distributed as follows: ${formatPercentage(dist.positive_percentage)} positive, ${formatPercentage(dist.neutral_percentage)} neutral, and ${formatPercentage(dist.negative_percentage)} negative. The most discussed stock is ${topStock?.ticker || 'N/A'} (${topStock?.company_name || 'N/A'}) with ${topStock?.count || 0} mentions.

Recent activity shows ${recentActivity.last_24h} predictions in the last 24 hours, ${recentActivity.last_7d} in the last 7 days, and ${recentActivity.last_30d} in the last 30 days.

The system has analyzed ${formatNumber(statistics.total_predictions)} records across ${statistics.total_stocks_analyzed} unique stocks, providing comprehensive coverage of market sentiment.`;
  };

  const sentimentTrendData = generateTrendData(20, 'up');
  const documentTrendData = generateTrendData(20, 'up');

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
              data={documentTrendData.map(v => v * statistics.total_predictions)} 
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

      {/* Correlation Overview Section */}
      {correlationOverview.length > 0 && (
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
            <CorrelationHeatmap
              data={correlationOverview.slice(0, 12)}
              onStockClick={handleStockClick}
            />
          </div>
        </ErrorBoundary>
      )}
    </div>
  );
};

export default Homepage;
