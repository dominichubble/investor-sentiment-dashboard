import React, { useState, useRef, useEffect } from 'react';
import './DropdownButton.css';

const imgPolygon1 = "https://www.figma.com/api/mcp/asset/ff420626-346a-48b3-9b6d-7bf7460bd5a6";

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
}

const DropdownButton: React.FC<DropdownButtonProps> = ({
  label,
  options,
  selectedOption,
  onSelect,
  className
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
    setIsOpen(!isOpen);
  };

  const handleOptionClick = (option: DropdownOption) => {
    onSelect(option);
    setIsOpen(false);
  };

  return (
    <div className={`dropdown-container ${className || ''}`} ref={dropdownRef}>
      <button
        className="dropdown-button"
        onClick={handleToggle}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
      >
        <p className="dropdown-text">
          <span className="dropdown-label">{label}:</span>
          <span className="dropdown-value"> {selectedOption.label}</span>
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
