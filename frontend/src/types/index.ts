export type SentimentValue = 'positive' | 'neutral' | 'negative';

export interface SentimentData {
  score: number;
  label: string;
  description: string;
}

export interface MetricCardData {
  id: string;
  title: string;
  value: string | number;
  description: string;
  trend?: number;
}

export interface AssetFilter {
  id: string;
  label: string;
  category: 'all' | 'etf' | 'crypto' | 'stock';
}

export interface SentimentBreakdown {
  positive: number;
  neutral: number;
  negative: number;
}

export interface GraphData {
  type: 'line' | 'bar' | 'breakdown';
  data: number[] | SentimentBreakdown;
}

export interface DashboardData {
  netSentiment: SentimentData;
  sentimentBreakdown: SentimentBreakdown;
  totalDocuments: number;
  activeSources: number;
  summaryText: string;
  confidence: number;
}
