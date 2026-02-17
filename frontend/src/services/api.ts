import axios from 'axios';
import type {
  CorrelationResponse,
  CorrelationOverviewItem,
  LagAnalysisResponse,
  TimeSeriesResponse,
  PriceHistoryResponse,
  StockInfoResponse,
  GrangerCausalityResponse,
  RollingCorrelationResponse,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface SentimentInfo {
  label: string;
  score: number;
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
  text: string;
  sentiment: SentimentInfo;
  source?: string;
  timestamp: string;
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
  async getStatistics(): Promise<StatisticsResponse> {
    const response = await api.get<StatisticsResponse>('/data/statistics');
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

  // --- Correlation endpoints ---

  async getCorrelation(
    ticker: string,
    params?: {
      period?: string;
      sentiment_metric?: string;
      price_metric?: string;
    }
  ): Promise<CorrelationResponse> {
    const response = await api.get<CorrelationResponse>(
      `/correlation/${ticker}`,
      { params }
    );
    return response.data;
  },

  async getCorrelationTimeseries(
    ticker: string,
    params?: { period?: string }
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
      sentiment_metric?: string;
    }
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
    }
  ): Promise<CorrelationOverviewItem[]> {
    const response = await api.get<CorrelationOverviewItem[]>(
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

  async getGrangerCausality(
    ticker: string,
    params?: {
      max_lag?: number;
      period?: string;
      sentiment_metric?: string;
    }
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
      sentiment_metric?: string;
      price_metric?: string;
    }
  ): Promise<RollingCorrelationResponse> {
    const response = await api.get<RollingCorrelationResponse>(
      `/correlation/${ticker}/rolling`,
      { params }
    );
    return response.data;
  },

  // Health check
  async healthCheck(): Promise<any> {
    const response = await axios.get(`${API_BASE_URL.replace('/api/v1', '')}/health`);
    return response.data;
  },
};

export default apiService;
