import React from 'react';
import { NavLink } from 'react-router-dom';
import DropdownButton, { DropdownOption } from '../DropdownButton';
import './Navbar.css';

interface NavbarProps {
  title?: string;
  subtitle?: string;
  /** Show links to market overview and stock analysis (default: true) */
  siteNav?: boolean;
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
  siteNav = true,
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

      {siteNav && (
        <div className="navbar-site-nav">
          <div className="navbar-nav" role="navigation" aria-label="Site sections">
            <NavLink
              to="/"
              end
              className={({ isActive }) =>
                `navbar-nav-item${isActive ? ' active' : ''}`
              }
            >
              Market overview
            </NavLink>
            <NavLink
              to="/analyze"
              className={({ isActive }) =>
                `navbar-nav-item${isActive ? ' active' : ''}`
              }
            >
              Stock analysis
            </NavLink>
            <NavLink
              to="/legal"
              className={({ isActive }) =>
                `navbar-nav-item${isActive ? ' active' : ''}`
              }
            >
              Legal
            </NavLink>
          </div>
        </div>
      )}

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
