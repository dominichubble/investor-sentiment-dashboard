import React from 'react';
import { Link } from 'react-router-dom';
import './SiteFooter.css';

export default function SiteFooter() {
  return (
    <footer className="site-footer" role="contentinfo">
      <div className="site-footer__inner">
        <p className="site-footer__disclaimer">
          Educational analytics only — not investment advice. Past sentiment and correlation do not predict
          returns. Verify material decisions with licensed professionals and primary sources.
        </p>
        <nav className="site-footer__nav" aria-label="Footer">
          <Link to="/legal" className="site-footer__link">
            Legal &amp; disclaimer
          </Link>
          <span className="site-footer__sep" aria-hidden>
            ·
          </span>
          <Link to="/" className="site-footer__link">
            Market overview
          </Link>
          <span className="site-footer__sep" aria-hidden>
            ·
          </span>
          <Link to="/analyze" className="site-footer__link">
            Stock analysis
          </Link>
          <span className="site-footer__sep" aria-hidden>
            ·
          </span>
          <Link to="/lime" className="site-footer__link">
            LIME explainability
          </Link>
        </nav>
      </div>
    </footer>
  );
}
