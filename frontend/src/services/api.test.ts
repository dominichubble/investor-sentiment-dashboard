import { describe, it, expect, vi, beforeEach } from 'vitest';
import axios from 'axios';
import { apiService } from './api';

vi.mock('axios', () => {
  const mockAxios = {
    create: vi.fn(() => mockAxios),
    get: vi.fn(),
    post: vi.fn(),
    defaults: { headers: { common: {} } },
    interceptors: {
      request: { use: vi.fn() },
      response: { use: vi.fn() },
    },
  };
  return { default: mockAxios };
});

const mockedAxios = axios as any;

describe('apiService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('getStatistics', () => {
    it('calls the correct endpoint', async () => {
      const mockData = {
        total_predictions: 100,
        total_stocks_analyzed: 10,
        sentiment_distribution: {
          positive: 40,
          negative: 30,
          neutral: 30,
          positive_percentage: 40,
          negative_percentage: 30,
          neutral_percentage: 30,
        },
        top_stocks: [],
        recent_activity: { last_24h: 5, last_7d: 30, last_30d: 100 },
        date_range: { earliest: null, latest: null },
      };

      mockedAxios.get.mockResolvedValueOnce({ data: mockData });
      const result = await apiService.getStatistics();
      expect(mockedAxios.get).toHaveBeenCalledWith('/data/statistics');
      expect(result).toEqual(mockData);
    });
  });

  describe('getCorrelation', () => {
    it('calls with correct ticker and params', async () => {
      const mockCorr = {
        ticker: 'AAPL',
        data_points: 30,
        pearson: { coefficient: 0.25, p_value: 0.01, significant: true, interpretation: 'Weak positive' },
        spearman: { coefficient: 0.22, p_value: 0.02, significant: true, interpretation: 'Weak positive' },
      };

      mockedAxios.get.mockResolvedValueOnce({ data: mockCorr });
      const result = await apiService.getCorrelation('AAPL', { period: '90d' });
      expect(mockedAxios.get).toHaveBeenCalledWith('/correlation/AAPL', { params: { period: '90d' } });
      expect(result.ticker).toBe('AAPL');
    });
  });

  describe('getGrangerCausality', () => {
    it('calls the granger endpoint', async () => {
      const mockGranger = {
        ticker: 'TSLA',
        max_lag: 5,
        summary: {
          sentiment_predicts_price: true,
          price_predicts_sentiment: false,
          interpretation: 'Sentiment Granger-causes price.',
        },
      };

      mockedAxios.get.mockResolvedValueOnce({ data: mockGranger });
      const result = await apiService.getGrangerCausality('TSLA', { max_lag: 5 });
      expect(mockedAxios.get).toHaveBeenCalledWith('/correlation/TSLA/granger', { params: { max_lag: 5 } });
      expect(result.ticker).toBe('TSLA');
    });
  });

  describe('getRollingCorrelation', () => {
    it('calls the rolling endpoint', async () => {
      const mockRolling = {
        ticker: 'NVDA',
        window: 14,
        data_points: 20,
        series: [{ date: '2026-01-01', correlation: 0.3, window_start: '2025-12-18' }],
      };

      mockedAxios.get.mockResolvedValueOnce({ data: mockRolling });
      const result = await apiService.getRollingCorrelation('NVDA', { window: 14 });
      expect(mockedAxios.get).toHaveBeenCalledWith('/correlation/NVDA/rolling', { params: { window: 14 } });
      expect(result.window).toBe(14);
    });
  });

  describe('getCorrelationOverview', () => {
    it('calls the overview endpoint with params', async () => {
      mockedAxios.get.mockResolvedValueOnce({
        data: {
          n_tickers_tested: 0,
          alpha_individual: 0.05,
          alpha_bonferroni: null,
          align_mode: 'same_day',
          market_adjustment: 'none',
          data_source: null,
          items: [],
        },
      });
      await apiService.getCorrelationOverview({ min_mentions: 5, period: '90d' });
      expect(mockedAxios.get).toHaveBeenCalledWith('/correlation/overview/all', {
        params: { min_mentions: 5, period: '90d' },
      });
    });
  });
});
