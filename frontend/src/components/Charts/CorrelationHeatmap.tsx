import React from 'react';
import type { CorrelationOverviewItem } from '../../types';

interface CorrelationHeatmapProps {
  data: CorrelationOverviewItem[];
  onStockClick?: (ticker: string) => void;
}

const CorrelationHeatmap: React.FC<CorrelationHeatmapProps> = ({
  data,
  onStockClick,
}) => {
  if (!data || data.length === 0) {
    return (
      <div style={{ padding: 40, textAlign: 'center', color: '#999' }}>
        No correlation data available. Ensure stocks have sufficient sentiment and price data.
      </div>
    );
  }

  const getColor = (r: number, significant: boolean): string => {
    if (!significant) return '#f5f5f5';
    const intensity = Math.min(Math.abs(r), 1);
    if (r > 0) {
      const g = Math.floor(120 + intensity * 80);
      return `rgba(122, ${g}, 134, ${0.2 + intensity * 0.6})`;
    } else {
      const rVal = Math.floor(180 + intensity * 40);
      return `rgba(${rVal}, 110, 104, ${0.2 + intensity * 0.6})`;
    }
  };

  const getTextColor = (r: number, significant: boolean): string => {
    if (!significant) return '#999';
    return Math.abs(r) > 0.5 ? 'white' : '#333';
  };

  return (
    <div style={{ overflowX: 'auto' }}>
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.min(data.length, 6)}, 1fr)`, gap: 8 }}>
        {data.map((item) => (
          <div
            key={item.ticker}
            onClick={() => onStockClick?.(item.ticker)}
            style={{
              background: getColor(item.pearson_r, item.significant),
              borderRadius: 8,
              padding: '16px 12px',
              textAlign: 'center',
              cursor: onStockClick ? 'pointer' : 'default',
              transition: 'transform 0.2s, box-shadow 0.2s',
              border: `1px solid ${item.significant ? '#e0e0e0' : '#eee'}`,
              minWidth: 120,
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-2px)';
              e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.1)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = 'none';
            }}
          >
            <div style={{
              fontWeight: 700,
              fontSize: 16,
              color: getTextColor(item.pearson_r, item.significant),
              marginBottom: 4,
            }}>
              {item.ticker}
            </div>
            <div style={{
              fontSize: 24,
              fontWeight: 700,
              color: getTextColor(item.pearson_r, item.significant),
              marginBottom: 4,
            }}>
              {item.pearson_r >= 0 ? '+' : ''}{item.pearson_r.toFixed(2)}
            </div>
            <div style={{
              fontSize: 11,
              color: getTextColor(item.pearson_r, item.significant),
              opacity: 0.8,
            }}>
              {item.mentions} mentions
            </div>
            <div style={{
              fontSize: 10,
              color: getTextColor(item.pearson_r, item.significant),
              opacity: 0.7,
              marginTop: 2,
            }}>
              {item.significant ? 'p < 0.05' : 'n.s.'}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default CorrelationHeatmap;
