import React from 'react';
import './MetricCard.css';

interface MetricCardProps {
  title: string;
  value: string | number;
  description: string;
  trend?: 'up' | 'down' | 'neutral';
  onClick?: () => void;
  className?: string;
  chart?: React.ReactNode;
}

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  description,
  trend = 'neutral',
  onClick,
  className,
  chart
}) => {
  const getValueColor = () => {
    if (typeof value === 'string' && value.startsWith('+')) {
      return '#6cdf7e'; // Positive (green)
    } else if (typeof value === 'string' && value.startsWith('-')) {
      return '#cb6e68'; // Negative (red)
    }
    return '#8e94a0'; // Neutral (gray)
  };

  return (
    <div 
      className={`metric-card ${onClick ? 'clickable' : ''} ${className || ''}`}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyPress={(e) => {
        if (onClick && (e.key === 'Enter' || e.key === ' ')) {
          onClick();
        }
      }}
    >
      <div className="metric-card-header">
        <h3 className="metric-card-title">{title}</h3>
        <div className="metric-card-divider" />
      </div>
      
      <div className="metric-card-content">
        <p className="metric-card-value" style={{ color: getValueColor() }}>
          {value}
        </p>
        <p className="metric-card-description">{description}</p>
        
        {chart && (
          <div className="metric-card-chart">
            {chart}
          </div>
        )}
      </div>

      {trend && (
        <div className={`metric-card-trend trend-${trend}`}>
          {trend === 'up' && '↑'}
          {trend === 'down' && '↓'}
          {trend === 'neutral' && '→'}
        </div>
      )}
    </div>
  );
};

export default MetricCard;
