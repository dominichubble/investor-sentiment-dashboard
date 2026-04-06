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
      <div className="navbar-inner">
        <div className="navbar-primary">
          <NavLink to="/" className="navbar-brand-link" end>
            <span className="navbar-logo-mark" aria-hidden />
            <span className="navbar-wordmark">
              <span className="navbar-wordmark__name">Sentiment Lab</span>
              <span className="navbar-wordmark__tag">Dissertation prototype</span>
            </span>
          </NavLink>

          {siteNav && (
            <div className="navbar-nav" role="navigation" aria-label="Site sections">
              <NavLink
                to="/"
                end
                className={({ isActive }) =>
                  `navbar-nav-item${isActive ? ' is-active' : ''}`
                }
              >
                Overview
              </NavLink>
              <NavLink
                to="/analyze"
                className={({ isActive }) =>
                  `navbar-nav-item${isActive ? ' is-active' : ''}`
                }
              >
                Stock analysis
              </NavLink>
              <NavLink
                to="/methodology"
                className={({ isActive }) =>
                  `navbar-nav-item${isActive ? ' is-active' : ''}`
                }
              >
                Methodology
              </NavLink>
              <NavLink
                to="/legal"
                className={({ isActive }) =>
                  `navbar-nav-item${isActive ? ' is-active' : ''}`
                }
              >
                Legal
              </NavLink>
            </div>
          )}
        </div>

        {(title || subtitle) && (
          <div className="navbar-context">
            {title && <h1 className="navbar-title">{title}</h1>}
            {subtitle && <p className="navbar-subtitle">{subtitle}</p>}
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
      </div>
    </nav>
  );
};

export default Navbar;
