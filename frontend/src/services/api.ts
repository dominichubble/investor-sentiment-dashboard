import axios from 'axios';
import type {
  CorrelationResponse,
  CorrelationOverviewResponse,
  CorrelationMethodologyParams,
  LagAnalysisResponse,
  TimeSeriesResponse,
  PriceHistoryResponse,
  StockInfoResponse,
  StockDataQualityResponse,
  GrangerCausalityResponse,
  RollingCorrelationResponse,
  OutOfSampleResponse,
} from '../types';

/** Ensure base URL ends with /api/v1 (common Vercel misconfig omits it). */
function resolveApiBaseUrl(): string {
  const raw = (import.meta.env.VITE_API_URL || '').trim() || 'http://localhost:8000/api/v1';
  const base = raw.replace(/\/+$/, '');
  if (base.endsWith('/api/v1')) return base;
  return `${base}/api/v1`;
}

const API_BASE_URL = resolveApiBaseUrl();

const apiKey = (import.meta.env.VITE_API_KEY || '').trim();

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
    ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
  },
});

export interface SentimentInfo {
  label: string;
  score: number;
}

export interface ExplanationToken {
  token: string;
  weight: number;
}

export interface TickerNarrativeResponse {
  narrative: string;
  cached: boolean;
  model: string;
  record_count: number;
  window_start: string | null;
  window_end: string | null;
  period_key: string;
  data_signature: string | null;
  error: string | null;
}

export interface ExplainSentimentResponse {
  text: string;
  prediction: {
    label: string;
    score: number;
    scores?: {
      positive: number;
      negative: number;
      neutral: number;
    };
  };
  tokens: ExplanationToken[];
  metadata: {
    method: string;
    num_features: number;
    num_samples: number;
    processing_time_ms: number;
    timestamp: string;
  };
}

export interface SentimentBreakdown {
  positive: number;
  negative: number;
  neutral: number;
  positive_percentage: number;
  negative_percentage: number;
  neutral_percentage: number;
}

export interface StockInfo {
  ticker: string;
  company_name: string;
  count: number;
  positive: number;
  negative: number;
  neutral: number;
}

export interface DailyTrendPoint {
  date: string;
  count: number;
  net_sentiment: number;
}

/** Per-channel totals for comparing Reddit vs news vs X */
export interface SourceSentimentBlock {
  total: number;
  positive: number;
  negative: number;
  neutral: number;
  positive_percentage: number;
  negative_percentage: number;
  neutral_percentage: number;
}

export interface SourceComparison {
  reddit: SourceSentimentBlock;
  news: SourceSentimentBlock;
  twitter: SourceSentimentBlock;
}

/** Daily cross-source disagreement (market-wide, all channels view) */
export interface SourceDisagreementDay {
  date: string;
  total_mentions: number;
  n_sources_active: number;
  disagreement_range: number | null;
  disagreement_std: number | null;
  net_by_source: Record<string, number>;
  counts_by_source: Record<string, number>;
}

export type SentimentSourceFilter = 'all' | 'reddit' | 'news' | 'twitter';

export interface StatisticsResponse {
  total_predictions: number;
  total_stocks_analyzed: number;
  sentiment_distribution: SentimentBreakdown;
  top_stocks: StockInfo[];
  recent_activity: {
    last_24h: number;
    last_7d: number;
    last_30d: number;
  };
  date_range: {
    earliest: string | null;
    latest: string | null;
  };
  daily_trend: DailyTrendPoint[];
  /** Present when no data_source filter is applied */
  source_comparison?: SourceComparison | null;
  /** Daily spread of net sentiment between Reddit, news, and X (all-sources view) */
  source_disagreement_trend?: SourceDisagreementDay[];
}

export interface TrendingStock {
  ticker: string;
  mentions: number;
}

export interface TrendingStocksResponse {
  trending: TrendingStock[];
  period_hours: number;
  total_stocks: number;
}

export interface PredictionRecord {
  id: string;
  record_type: string;
  text: string;
  sentiment: SentimentInfo;
  source?: string;
  published_at: string;
  metadata?: any;
}

export interface PredictionsResponse {
  predictions: PredictionRecord[];
  total: number;
  page: number;
  page_size: number;
  has_more: boolean;
}

export interface StockSentimentResponse {
  ticker: string;
  total_mentions: number;
  average_score: number;
  sentiment_distribution: {
    positive: number;
    negative: number;
    neutral: number;
  };
  records?: any[];
}

// API Service
export const apiService = {
  // Data endpoints
  async getStatistics(params?: {
    days?: number;
    /** reddit | news | twitter | x — omit for all sources */
    data_source?: string;
  }): Promise<StatisticsResponse> {
    const response = await api.get<StatisticsResponse>('/data/statistics', { params });
    return response.data;
  },

  async getPredictions(params?: {
    page?: number;
    page_size?: number;
    source?: string;
    sentiment?: string;
    start_date?: string;
    end_date?: string;
  }): Promise<PredictionsResponse> {
    const response = await api.get<PredictionsResponse>('/data/predictions', { params });
    return response.data;
  },

  async getStockSentiment(ticker: string, params?: {
    limit?: number;
    start_date?: string;
  }): Promise<any> {
    const response = await api.get(`/data/stocks/${ticker}/sentiment`, { params });
    return response.data;
  },

  // Stock endpoints
  async getTrendingStocks(params?: {
    period?: string;
    min_mentions?: number;
    limit?: number;
  }): Promise<TrendingStocksResponse> {
    const response = await api.get<TrendingStocksResponse>('/stocks/trending', { params });
    return response.data;
  },

  async getStockSentimentAggregated(
    ticker: string,
    params?: {
      start_date?: string;
      end_date?: string;
      source?: string;
      include_records?: boolean;
    }
  ): Promise<StockSentimentResponse> {
    const response = await api.get<StockSentimentResponse>(
      `/stocks/${ticker}/sentiment`,
      { params }
    );
    return response.data;
  },

  async compareStocks(
    tickers: string[],
    params?: {
      start_date?: string;
      end_date?: string;
    }
  ): Promise<any> {
    const queryString = tickers.map(t => `tickers=${t}`).join('&');
    const response = await api.post(`/stocks/compare?${queryString}`, null, { params });
    return response.data;
  },

  async getStockStatistics(): Promise<any> {
    const response = await api.get('/stocks/statistics');
    return response.data;
  },

  // Sentiment analysis endpoint
  async analyzeSentiment(text: string, options?: any): Promise<any> {
    const response = await api.post('/sentiment/analyze', { text, options });
    return response.data;
  },

  async batchAnalyzeSentiment(texts: string[], options?: any): Promise<any> {
    const response = await api.post('/sentiment/batch', { texts, options });
    return response.data;
  },

  async explainSentiment(
    text: string,
    options?: any,
    signal?: AbortSignal,
  ): Promise<ExplainSentimentResponse> {
    const response = await api.post<ExplainSentimentResponse>(
      '/sentiment/explain',
      { text, options },
      { timeout: 120_000, signal },   // 120 s budget for CPU-heavy LIME
    );
    return response.data;
  },

  /** Grounded AI summary of ticker sentiment (Groq); same date window as correlation. */
  async getTickerSentimentNarrative(
    ticker: string,
    params: {
      period?: string;
      start_date?: string;
      end_date?: string;
      force_refresh?: boolean;
    },
    signal?: AbortSignal,
  ): Promise<TickerNarrativeResponse> {
    const response = await api.get<TickerNarrativeResponse>(
      `/sentiment/ticker-narrative/${encodeURIComponent(ticker)}`,
      {
        params: {
          period: params.period ?? '90d',
          start_date: params.start_date,
          end_date: params.end_date,
          force_refresh: params.force_refresh ?? false,
        },
        timeout: 90_000,
        signal,
      },
    );
    return response.data;
  },

  // --- Correlation endpoints ---

  async getCorrelation(
    ticker: string,
    params?: {
      period?: string;
      start_date?: string;
      end_date?: string;
      sentiment_metric?: string;
      price_metric?: string | null;
      trailing_days?: number;
    } & CorrelationMethodologyParams
  ): Promise<CorrelationResponse> {
    const response = await api.get<CorrelationResponse>(
      `/correlation/${ticker}`,
      { params }
    );
    return response.data;
  },

  async getCorrelationTimeseries(
    ticker: string,
    params?: {
      period?: string;
      start_date?: string;
      end_date?: string;
      trailing_days?: number;
    } & CorrelationMethodologyParams
  ): Promise<TimeSeriesResponse> {
    const response = await api.get<TimeSeriesResponse>(
      `/correlation/${ticker}/timeseries`,
      { params }
    );
    return response.data;
  },

  async getLagAnalysis(
    ticker: string,
    params?: {
      max_lag_days?: number;
      period?: string;
      start_date?: string;
      end_date?: string;
      sentiment_metric?: string;
      trailing_days?: number;
    } & CorrelationMethodologyParams
  ): Promise<LagAnalysisResponse> {
    const response = await api.get<LagAnalysisResponse>(
      `/correlation/${ticker}/lag-analysis`,
      { params }
    );
    return response.data;
  },

  async getCorrelationOverview(
    params?: {
      min_mentions?: number;
      period?: string;
    } & CorrelationMethodologyParams
  ): Promise<CorrelationOverviewResponse> {
    const response = await api.get<CorrelationOverviewResponse>(
      '/correlation/overview/all',
      { params }
    );
    return response.data;
  },

  async getPriceHistory(
    ticker: string,
    params?: { period?: string }
  ): Promise<PriceHistoryResponse> {
    const response = await api.get<PriceHistoryResponse>(
      `/correlation/${ticker}/price-history`,
      { params }
    );
    return response.data;
  },

  async getStockInfo(ticker: string): Promise<StockInfoResponse> {
    const response = await api.get<StockInfoResponse>(
      `/correlation/${ticker}/info`
    );
    return response.data;
  },

  async getStockDataQuality(
    ticker: string,
    params?: {
      period?: string;
      start_date?: string;
      end_date?: string;
      data_source?: string;
    },
  ): Promise<StockDataQualityResponse> {
    const response = await api.get<StockDataQualityResponse>(
      `/data/stock-quality/${encodeURIComponent(ticker)}`,
      { params },
    );
    return response.data;
  },

  async getGrangerCausality(
    ticker: string,
    params?: {
      max_lag?: number;
      period?: string;
      sentiment_metric?: string;
      start_date?: string;
      end_date?: string;
      trailing_days?: number;
    } & CorrelationMethodologyParams
  ): Promise<GrangerCausalityResponse> {
    const response = await api.get<GrangerCausalityResponse>(
      `/correlation/${ticker}/granger`,
      { params }
    );
    return response.data;
  },

  async getRollingCorrelation(
    ticker: string,
    params?: {
      window?: number;
      period?: string;
      start_date?: string;
      end_date?: string;
      sentiment_metric?: string;
      price_metric?: string | null;
      trailing_days?: number;
    } & CorrelationMethodologyParams
  ): Promise<RollingCorrelationResponse> {
    const response = await api.get<RollingCorrelationResponse>(
      `/correlation/${ticker}/rolling`,
      { params }
    );
    return response.data;
  },

  async getOutOfSampleCorrelation(
    ticker: string,
    params?: {
      period?: string;
      sentiment_metric?: string;
      price_metric?: string | null;
      train_ratio?: number;
      start_date?: string;
      end_date?: string;
      trailing_days?: number;
    } & CorrelationMethodologyParams,
  ): Promise<OutOfSampleResponse> {
    const response = await api.get<OutOfSampleResponse>(
      `/correlation/${ticker}/out-of-sample`,
      { params },
    );
    return response.data;
  },

  // Health check
  async healthCheck(): Promise<any> {
    const base = API_BASE_URL.replace(/\/api\/v1\/?$/, '');
    const response = await axios.get(`${base}/health`);
    return response.data;
  },
};

export default apiService;
