import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { DashboardProvider, useDashboard } from './DashboardContext';
import { apiService } from '../services/api';

vi.mock('../services/api', () => ({
  apiService: {
    getStatistics: vi.fn(),
    getCorrelationOverview: vi.fn(),
  },
}));

function TestConsumer() {
  const { statistics, correlationOverview, isLoading, error } = useDashboard();

  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div>
      <div data-testid="total-predictions">
        {statistics?.total_predictions ?? 'none'}
      </div>
      <div data-testid="overview-count">
        {correlationOverview.length}
      </div>
    </div>
  );
}

describe('DashboardContext', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('provides statistics data to consumers', async () => {
    const mockStats = {
      total_predictions: 500,
      total_stocks_analyzed: 25,
      sentiment_distribution: {
        positive: 200, negative: 150, neutral: 150,
        positive_percentage: 40, negative_percentage: 30, neutral_percentage: 30,
      },
      top_stocks: [],
      recent_activity: { last_24h: 10, last_7d: 50, last_30d: 200 },
      date_range: { earliest: null, latest: null },
    };

    (apiService.getStatistics as any).mockResolvedValueOnce(mockStats);
    (apiService.getCorrelationOverview as any).mockResolvedValueOnce([]);

    render(
      <DashboardProvider>
        <TestConsumer />
      </DashboardProvider>
    );

    expect(screen.getByText('Loading...')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByTestId('total-predictions')).toHaveTextContent('500');
    });
  });

  it('provides correlation overview data', async () => {
    (apiService.getStatistics as any).mockResolvedValueOnce({
      total_predictions: 100,
      total_stocks_analyzed: 5,
      sentiment_distribution: {
        positive: 40, negative: 30, neutral: 30,
        positive_percentage: 40, negative_percentage: 30, neutral_percentage: 30,
      },
      top_stocks: [],
      recent_activity: { last_24h: 1, last_7d: 10, last_30d: 50 },
      date_range: { earliest: null, latest: null },
    });
    (apiService.getCorrelationOverview as any).mockResolvedValueOnce([
      { ticker: 'AAPL', mentions: 50, data_points: 30, pearson_r: 0.35, pearson_p: 0.01, significant: true, interpretation: 'Moderate positive', spearman_r: 0.32 },
      { ticker: 'TSLA', mentions: 40, data_points: 25, pearson_r: -0.2, pearson_p: 0.1, significant: false, interpretation: 'Weak negative', spearman_r: -0.18 },
    ]);

    render(
      <DashboardProvider>
        <TestConsumer />
      </DashboardProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('overview-count')).toHaveTextContent('2');
    });
  });

  it('handles API errors gracefully', async () => {
    (apiService.getStatistics as any).mockRejectedValueOnce(new Error('Network error'));
    (apiService.getCorrelationOverview as any).mockRejectedValueOnce(new Error('Network error'));

    render(
      <DashboardProvider>
        <TestConsumer />
      </DashboardProvider>
    );

    await waitFor(() => {
      expect(screen.getByText(/Error:/)).toBeInTheDocument();
    });
  });
});
