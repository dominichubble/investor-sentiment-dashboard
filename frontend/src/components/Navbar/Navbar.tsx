import React from 'react';
import DropdownButton, { DropdownOption } from '../DropdownButton';
import './Navbar.css';

interface NavbarProps {
  title?: string;
  assetOptions: DropdownOption[];
  selectedAsset: DropdownOption;
  onAssetChange: (option: DropdownOption) => void;
  timeframeOptions: DropdownOption[];
  selectedTimeframe: DropdownOption;
  onTimeframeChange: (option: DropdownOption) => void;
  className?: string;
}

const Navbar: React.FC<NavbarProps> = ({
  title = "COC251 Sentiment Analysis | Overview",
  assetOptions,
  selectedAsset,
  onAssetChange,
  timeframeOptions,
  selectedTimeframe,
  onTimeframeChange,
  className
}) => {
  return (
    <nav className={`navbar ${className || ''}`}>
      <h1 className="navbar-title">{title}</h1>
      <div className="navbar-buttons">
        <DropdownButton
          label="Asset Selector"
          options={assetOptions}
          selectedOption={selectedAsset}
          onSelect={onAssetChange}
        />
        <DropdownButton
          label="Timeframe"
          options={timeframeOptions}
          selectedOption={selectedTimeframe}
          onSelect={onTimeframeChange}
        />
      </div>
    </nav>
  );
};

export default Navbar;
