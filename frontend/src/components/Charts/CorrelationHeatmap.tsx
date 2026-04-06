import React from 'react';
import type { CorrelationOverviewItem } from '../../types';
import { formatDecimalDisplay, formatIntegerDisplay } from '../../utils/formatDisplay';
import './CorrelationHeatmap.css';

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
      <div className="heatmap-empty">
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
    <div className="heatmap-container">
      <div
        className="heatmap-grid"
        style={{ gridTemplateColumns: `repeat(${Math.min(data.length, 6)}, 1fr)` }}
      >
        {data.map((item) => {
          const textColor = getTextColor(item.pearson_r, item.significant);
          return (
            <div
              key={item.ticker}
              className="heatmap-tile"
              onClick={() => onStockClick?.(item.ticker)}
              style={{
                background: getColor(item.pearson_r, item.significant),
                cursor: onStockClick ? 'pointer' : 'default',
                borderColor: item.significant ? '#e0e0e0' : '#eee',
              }}
            >
              <div className="heatmap-ticker" style={{ color: textColor }}>
                {item.ticker}
              </div>
              <div className="heatmap-value" style={{ color: textColor }}>
                {item.pearson_r >= 0 ? '+' : ''}
                {formatDecimalDisplay(item.pearson_r, 2)}
              </div>
              <div className="heatmap-mentions" style={{ color: textColor }}>
                {formatIntegerDisplay(item.mentions)} mentions
              </div>
              <div className="heatmap-significance" style={{ color: textColor }}>
                {item.significant ? 'p < 0.05' : 'n.s.'}
                {item.significant_bonferroni ? ' · Bonf.' : ''}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default CorrelationHeatmap;
