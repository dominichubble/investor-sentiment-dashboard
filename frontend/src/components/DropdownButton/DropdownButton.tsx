import React, { useState, useRef, useEffect } from 'react';
import './DropdownButton.css';

const imgPolygon1 =
  "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath d='M5 6L0 0h10z' fill='%23555'/%3E%3C/svg%3E";

export interface DropdownOption {
  id: string;
  label: string;
  value: string;
}

interface DropdownButtonProps {
  label: string;
  options: DropdownOption[];
  selectedOption: DropdownOption;
  onSelect: (option: DropdownOption) => void;
  className?: string;
  disabled?: boolean;
  disabledMessage?: string;
}

const DropdownButton: React.FC<DropdownButtonProps> = ({
  label,
  options,
  selectedOption,
  onSelect,
  className,
  disabled = false,
  disabledMessage,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleToggle = () => {
    if (!disabled) setIsOpen(!isOpen);
  };

  const handleOptionClick = (option: DropdownOption) => {
    onSelect(option);
    setIsOpen(false);
  };

  return (
    <div className={`dropdown-container ${disabled ? 'disabled' : ''} ${className || ''}`} ref={dropdownRef}>
      <button
        className="dropdown-button"
        onClick={handleToggle}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        disabled={disabled}
        title={disabled ? disabledMessage : undefined}
      >
        <p className="dropdown-text">
          <span className="dropdown-label">{label}:</span>
          <span className="dropdown-value"> {selectedOption.label}</span>
          {disabled && disabledMessage && (
            <span className="dropdown-badge">{disabledMessage}</span>
          )}
        </p>
        <div className="dropdown-icon">
          <div className={`polygon-wrapper ${isOpen ? 'open' : ''}`}>
            <img alt="Dropdown arrow" src={imgPolygon1} />
          </div>
        </div>
      </button>

      {isOpen && (
        <div className="dropdown-menu" role="listbox">
          {options.map((option) => (
            <button
              key={option.id}
              className={`dropdown-menu-item ${option.id === selectedOption.id ? 'selected' : ''}`}
              onClick={() => handleOptionClick(option)}
              role="option"
              aria-selected={option.id === selectedOption.id}
            >
              {option.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default DropdownButton;
