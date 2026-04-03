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
  /** Backend trailing window used for net_sentiment correlations (1 = same day). */
  trailing_days?: number;
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
  trailing_days?: number;
  error?: string;
}

export interface TimeSeriesPoint {
  date: string;
  close: number;
  returns: number | null;
  avg_sentiment_score: number;
  net_sentiment: number;
  /** Causal rolling mean of net_sentiment; window length in TimeSeriesResponse.trailing_days. */
  trailing_net_sentiment: number;
  mention_count: number;
  positive_ratio: number;
  negative_ratio: number;
  neutral_ratio: number;
}

export interface TimeSeriesResponse {
  ticker: string;
  data_points: number;
  trailing_days: number;
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

/** Per-ticker sentiment data quality (same window as correlation / AI narrative). */
export interface DataQualityFlag {
  id: string;
  severity: string;
  title: string;
  detail: string;
}

export interface StockDataQualityResponse {
  ticker: string;
  window_start: string | null;
  window_end: string | null;
  calendar_days: number;
  days_with_mentions: number;
  calendar_coverage: number;
  longest_gap_days: number;
  total_mentions: number;
  by_label: Record<string, number>;
  label_shares: Record<string, number>;
  by_channel: Record<string, number>;
  confidence_score: number;
  confidence_label: string;
  flags: DataQualityFlag[];
  error?: string | null;
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
  trailing_days?: number;
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
  trailing_days?: number;
  error?: string;
}
