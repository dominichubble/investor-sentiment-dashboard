import React from 'react';
import DropdownButton, { DropdownOption } from '../DropdownButton';
import './Navbar.css';

interface NavbarProps {
  title?: string;
  subtitle?: string;
  assetOptions?: DropdownOption[];
  selectedAsset?: DropdownOption;
  onAssetChange?: (option: DropdownOption) => void;
  assetDisabled?: boolean;
  assetDisabledMessage?: string;
  timeframeOptions?: DropdownOption[];
  selectedTimeframe?: DropdownOption;
  onTimeframeChange?: (option: DropdownOption) => void;
  className?: string;
}

const Navbar: React.FC<NavbarProps> = ({
  title = 'Investor Sentiment — Correlation Analysis',
  subtitle,
  assetOptions,
  selectedAsset,
  onAssetChange,
  assetDisabled = false,
  assetDisabledMessage,
  timeframeOptions,
  selectedTimeframe,
  onTimeframeChange,
  className,
}) => {
  return (
    <nav className={`navbar ${className || ''}`} aria-label="Primary">
      <div className="navbar-top">
        <div className="navbar-brand-row">
          <div className="navbar-logo-mark" aria-hidden />
          <div className="navbar-brand">
            <p className="navbar-eyebrow">Investor sentiment dashboard</p>
            <h1 className="navbar-title">{title}</h1>
            {subtitle && <p className="navbar-subtitle">{subtitle}</p>}
          </div>
        </div>
      </div>

      {assetOptions && selectedAsset && onAssetChange && timeframeOptions && selectedTimeframe && onTimeframeChange && (
        <div className="navbar-filters">
          <DropdownButton
            label="Asset Selector"
            options={assetOptions}
            selectedOption={selectedAsset}
            onSelect={onAssetChange}
            disabled={assetDisabled}
            disabledMessage={assetDisabledMessage}
          />
          <DropdownButton
            label="Timeframe"
            options={timeframeOptions}
            selectedOption={selectedTimeframe}
            onSelect={onTimeframeChange}
          />
        </div>
      )}
    </nav>
  );
};

export default Navbar;
