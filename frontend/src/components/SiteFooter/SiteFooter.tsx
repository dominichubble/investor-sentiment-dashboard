import { Link } from 'react-router-dom';
import { BrandMark } from '../Brand';
import './SiteFooter.css';

export default function SiteFooter() {
  return (
    <footer className="site-footer" role="contentinfo">
      <div className="site-footer__inner">
        <div className="site-footer__brand">
          <BrandMark size={32} className="site-footer__mark" />
          <div className="site-footer__brand-copy">
            <span className="site-footer__brand-name">Sentiment Lab</span>
            <span className="site-footer__brand-tag">Market-wide &amp; per-ticker sentiment analytics</span>
          </div>
        </div>
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
          <span className="site-footer__sep" aria-hidden>
            ·
          </span>
          <Link to="/methodology" className="site-footer__link">
            Methodology
          </Link>
        </nav>
      </div>
    </footer>
  );
}
