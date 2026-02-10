import React, { useState, useEffect } from 'react';
import Navbar from '../../components/Navbar';
import MetricCard from '../../components/MetricCard';
import SummaryCard from '../../components/SummaryCard';
import { MiniLineChart, SentimentBarChart, MiniAreaChart } from '../../components/Charts';
import { DropdownOption } from '../../components/DropdownButton';
import { DashboardData } from '../../types';
import './Homepage.css';

// Mock data - replace with actual API calls
const mockDashboardData: DashboardData = {
  netSentiment: {
    score: 0.38,
    label: '+0.38',
    description: 'Mildly Positive (Normalized -1 to +1)'
  },
  sentimentBreakdown: {
    positive: 42,
    neutral: 38,
    negative: 20
  },
  totalDocuments: 12847,
  activeSources: 3,
  summaryText: `Over the past 7 days, aggregated sentiment across all asset classes has remained moderately positive (+0.38).

Positive sentiment is primarily driven by discussions surrounding Bitcoin ETF approvals on Twitter/X and select technology stock earnings in financial news. Conversely, inflation concerns and regulatory uncertainty continue to fuel negative sentiment in broader market discussions.

Neutral sentiment remains dominant in general market commentary.

The model exhibits high confidence (92.4%) across most classifications.`,
  confidence: 92.4
};

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
  const [selectedAsset, setSelectedAsset] = useState<DropdownOption>(assetOptions[0]);
  const [selectedTimeframe, setSelectedTimeframe] = useState<DropdownOption>(timeframeOptions[0]);
  const [dashboardData, setDashboardData] = useState<DashboardData>(mockDashboardData);
  const [isLoading, setIsLoading] = useState(false);

  // Fetch data when filters change
  useEffect(() => {
    fetchDashboardData();
  }, [selectedAsset, selectedTimeframe]);

  const fetchDashboardData = async () => {
    setIsLoading(true);
    try {
      // TODO: Replace with actual API call
      // const response = await fetch(`/api/sentiment?asset=${selectedAsset.value}&timeframe=${selectedTimeframe.value}`);
      // const data = await response.json();
      // setDashboardData(data);
      
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 500));
      setDashboardData(mockDashboardData);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAssetChange = (option: DropdownOption) => {
    setSelectedAsset(option);
  };

  const handleTimeframeChange = (option: DropdownOption) => {
    setSelectedTimeframe(option);
  };

  const handleCardClick = (cardType: string) => {
    console.log(`Card clicked: ${cardType}`);
    // TODO: Navigate to detailed view or open modal
  };

  const formatNumber = (num: number): string => {
    return num.toLocaleString();
  };

  // Generate mock trend data for visualizations
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

  const sentimentTrendData = generateTrendData(20, 'up');
  const documentTrendData = generateTrendData(20, 'up');

  return (
    <div className={`homepage ${isLoading ? 'loading' : ''}`}>
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
          value={dashboardData.netSentiment.label}
          description={dashboardData.netSentiment.description}
          trend={dashboardData.netSentiment.score > 0 ? 'up' : dashboardData.netSentiment.score < 0 ? 'down' : 'neutral'}
          onClick={() => handleCardClick('net-sentiment')}
          chart={
            <MiniLineChart 
              data={sentimentTrendData} 
              color="#6cdf7e"
            />
          }
        />
        
        <MetricCard
          title="SENTIMENT BREAKDOWN"
          value={`${dashboardData.sentimentBreakdown.positive}%`}
          description="Distribution across sentiment categories"
          onClick={() => handleCardClick('sentiment-breakdown')}
          chart={
            <SentimentBarChart
              positive={dashboardData.sentimentBreakdown.positive}
              neutral={dashboardData.sentimentBreakdown.neutral}
              negative={dashboardData.sentimentBreakdown.negative}
            />
          }
        />
        
        <MetricCard
          title="TOTAL DOCUMENTS ANALYSED"
          value={formatNumber(dashboardData.totalDocuments)}
          description={`From ${dashboardData.activeSources} active data sources`}
          onClick={() => handleCardClick('total-documents')}
          chart={
            <MiniAreaChart 
              data={documentTrendData.map(v => v * 15000)} 
              color="#5c7cfa"
            />
          }
        />
        
        <MetricCard
          title="ACTIVE DATA SOURCES"
          value={dashboardData.activeSources}
          description="Twitter/X, Reddit, Financial News"
          onClick={() => handleCardClick('active-sources')}
        />
      </div>

      <SummaryCard
        title="DOMINATED SENTIMENT TRENDS (SUMMARY)"
        summary={dashboardData.summaryText}
        embeddedCardValue={dashboardData.netSentiment.label}
        embeddedCardDescription={`Confidence: ${dashboardData.confidence}%`}
        embeddedCardChart={
          <SentimentBarChart
            positive={dashboardData.sentimentBreakdown.positive}
            neutral={dashboardData.sentimentBreakdown.neutral}
            negative={dashboardData.sentimentBreakdown.negative}
            height={100}
          />
        }
      />

      {isLoading && (
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
        </div>
      )}
    </div>
  );
};

export default Homepage;
