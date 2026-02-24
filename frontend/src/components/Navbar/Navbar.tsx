import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import DropdownButton, { DropdownOption } from '../DropdownButton';
import './Navbar.css';

interface NavbarProps {
  title?: string;
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
  title = "COC251 Sentiment Analysis",
  assetOptions,
  selectedAsset,
  onAssetChange,
  assetDisabled = false,
  assetDisabledMessage,
  timeframeOptions,
  selectedTimeframe,
  onTimeframeChange,
  className
}) => {
  const navigate = useNavigate();
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Overview' },
    { path: '/correlation', label: 'Correlation Analysis' },
  ];

  return (
    <nav className={`navbar ${className || ''}`}>
      <div className="navbar-top">
        <h1 className="navbar-title">{title}</h1>
        <div className="navbar-nav">
          {navItems.map(item => (
            <button
              key={item.path}
              className={`navbar-nav-item ${location.pathname === item.path ? 'active' : ''}`}
              onClick={() => navigate(item.path)}
            >
              {item.label}
            </button>
          ))}
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
