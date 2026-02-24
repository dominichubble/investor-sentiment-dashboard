import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import { apiService, StatisticsResponse } from '../services/api';
import type { CorrelationOverviewItem } from '../types';

interface DashboardState {
  statistics: StatisticsResponse | null;
  correlationOverview: CorrelationOverviewItem[];
  selectedAssetType: string;
  selectedTimeframe: string;
  isLoading: boolean;
  error: string | null;
}

interface DashboardContextType extends DashboardState {
  setSelectedAssetType: (assetType: string) => void;
  setSelectedTimeframe: (timeframe: string) => void;
  refreshData: () => Promise<void>;
}

const DashboardContext = createContext<DashboardContextType | undefined>(undefined);

export function DashboardProvider({ children }: { children: ReactNode }) {
  const [statistics, setStatistics] = useState<StatisticsResponse | null>(null);
  const [correlationOverview, setCorrelationOverview] = useState<CorrelationOverviewItem[]>([]);
  const [selectedAssetType, setSelectedAssetType] = useState('all');
  const [selectedTimeframe, setSelectedTimeframe] = useState('all');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      // Convert timeframe to days param (or undefined for 'all')
      const daysParam = selectedTimeframe === 'all' ? undefined : Number(selectedTimeframe);
      const corrPeriod = selectedTimeframe === 'all' ? '365d' : `${selectedTimeframe}d`;

      const [stats, corrOverview] = await Promise.allSettled([
        apiService.getStatistics(daysParam ? { days: daysParam } : undefined),
        apiService.getCorrelationOverview({ min_mentions: 2, period: corrPeriod }),
      ]);

      if (stats.status === 'fulfilled') {
        setStatistics(stats.value);
      } else {
        setError('Failed to load statistics.');
      }

      if (corrOverview.status === 'fulfilled') {
        setCorrelationOverview(corrOverview.value);
      }
    } catch (err: any) {
      setError(
        err.response?.data?.detail ||
        'Failed to load data. Please ensure the backend is running.'
      );
    } finally {
      setIsLoading(false);
    }
  }, [selectedTimeframe]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const value: DashboardContextType = {
    statistics,
    correlationOverview,
    selectedAssetType,
    selectedTimeframe,
    isLoading,
    error,
    setSelectedAssetType,
    setSelectedTimeframe,
    refreshData: fetchData,
  };

  return (
    <DashboardContext.Provider value={value}>
      {children}
    </DashboardContext.Provider>
  );
}

export function useDashboard(): DashboardContextType {
  const context = useContext(DashboardContext);
  if (!context) {
    throw new Error('useDashboard must be used within a DashboardProvider');
  }
  return context;
}

export default DashboardContext;
