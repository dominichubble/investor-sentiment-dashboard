import React from 'react';
import MetricCard from '../MetricCard';
import './SummaryCard.css';

interface SummaryCardProps {
  title: string;
  summary: string;
  embeddedCardTitle?: string;
  embeddedCardValue?: string;
  embeddedCardDescription?: string;
  embeddedCardChart?: React.ReactNode;
  className?: string;
}

const SummaryCard: React.FC<SummaryCardProps> = ({
  title,
  summary,
  embeddedCardTitle = "SENTIMENT DISTRIBUTION",
  embeddedCardValue = "+0.38",
  embeddedCardDescription = "Mildly Positive (Normalized -1 to +1)",
  embeddedCardChart,
  className
}) => {
  return (
    <div className={`summary-card ${className || ''}`}>
      <h2 className="summary-card-title">{title}</h2>
      
      <div className="summary-card-divider" />
      
      <div className="summary-card-content">
        <div className="summary-card-text">
          {summary.split('\n\n').map((paragraph, index) => (
            <p key={index}>{paragraph}</p>
          ))}
        </div>
        
        <div className="summary-card-embedded">
          <MetricCard
            title={embeddedCardTitle}
            value={embeddedCardValue}
            description={embeddedCardDescription}
            chart={embeddedCardChart}
          />
        </div>
      </div>
    </div>
  );
};

export default SummaryCard;
