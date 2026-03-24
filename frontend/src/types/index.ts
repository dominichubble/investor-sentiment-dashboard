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

// --- Correlation Types ---

export interface CorrelationResult {
  coefficient: number;
  p_value: number;
  significant: boolean;
  interpretation: string;
}

export interface CorrelationResponse {
  ticker: string;
  data_points: number;
  period?: string;
  sentiment_metric?: string;
  price_metric?: string;
  pearson: CorrelationResult | null;
  spearman: CorrelationResult | null;
  error?: string;
}

export interface LagResult {
  lag_days: number;
  data_points: number;
  pearson_r: number | null;
  p_value: number | null;
  significant?: boolean;
  description: string;
}

export interface LagAnalysisResponse {
  ticker: string;
  max_lag_days: number;
  lags: LagResult[];
  best_lag: LagResult | null;
  error?: string;
}

export interface TimeSeriesPoint {
  date: string;
  close: number;
  returns: number | null;
  avg_sentiment_score: number;
  net_sentiment: number;
  mention_count: number;
  positive_ratio: number;
  negative_ratio: number;
  neutral_ratio: number;
}

export interface TimeSeriesResponse {
  ticker: string;
  data_points: number;
  series: TimeSeriesPoint[];
}

export interface CorrelationOverviewItem {
  ticker: string;
  mentions: number;
  data_points: number;
  pearson_r: number;
  pearson_p: number;
  significant: boolean;
  interpretation: string;
  spearman_r: number;
}

export interface PriceHistoryPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  returns: number | null;
}

export interface PriceHistoryResponse {
  ticker: string;
  data_points: number;
  series: PriceHistoryPoint[];
}

export interface StockInfoResponse {
  ticker: string;
  name: string;
  sector?: string;
  industry?: string;
  market_cap?: number;
  currency?: string;
}

// --- Granger Causality Types ---

export interface GrangerLagResult {
  lag: number;
  f_statistic: number;
  p_value: number;
  significant: boolean;
}

export interface GrangerSummary {
  sentiment_predicts_price: boolean;
  price_predicts_sentiment: boolean;
  best_sentiment_to_price_lag: GrangerLagResult | null;
  best_price_to_sentiment_lag: GrangerLagResult | null;
  interpretation: string;
}

export interface GrangerCausalityResponse {
  ticker: string;
  max_lag?: number;
  data_points?: number;
  sentiment_to_price?: GrangerLagResult[];
  price_to_sentiment?: GrangerLagResult[];
  summary?: GrangerSummary;
  error?: string;
}

// --- Rolling Correlation Types ---

export interface RollingCorrelationPoint {
  date: string;
  correlation: number;
  window_start: string;
}

export interface RollingCorrelationStats {
  mean_correlation: number;
  std_correlation: number;
  min_correlation: number;
  max_correlation: number;
  periods_positive: number;
  periods_negative: number;
}

export interface RollingCorrelationResponse {
  ticker: string;
  window: number;
  period?: string;
  data_points: number;
  series: RollingCorrelationPoint[];
  statistics?: RollingCorrelationStats;
  error?: string;
}
