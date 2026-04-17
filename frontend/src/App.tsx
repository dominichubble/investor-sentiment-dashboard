import { useEffect } from 'react';
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
  Link,
  useSearchParams,
  useLocation,
} from 'react-router-dom';
import MarketOverview from './pages/MarketOverview/MarketOverview';
import StockAnalysis from './pages/StockAnalysis';
import Legal from './pages/Legal';
import Methodology from './pages/Methodology';
import LimeExplain from './pages/LimeExplain';
import SiteFooter from './components/SiteFooter';
import { BrandMark } from './components/Brand';
import { ErrorBoundary } from './components/ErrorBoundary';
import './App.css';

/** Legacy links to `/?ticker=…` (and related query params) go to stock analysis. */
function RootRoute() {
  const [searchParams] = useSearchParams();
  const ticker = searchParams.get('ticker')?.trim();
  if (ticker) {
    const q = searchParams.toString();
    return <Navigate to={q ? `/analyze?${q}` : '/analyze'} replace />;
  }
  return <MarketOverview />;
}

/**
 * Chrome “Capture full size screenshot” often drops or clips headers that use
 * `position: sticky` plus full-bleed `100vw` margins. Append `?capture=1` (keep
 * it when navigating) so the navbar sits in normal flow for that capture only.
 */
function CaptureLayoutSync() {
  const { search } = useLocation();
  useEffect(() => {
    const on = new URLSearchParams(search).get('capture') === '1';
    const el = document.documentElement;
    if (on) {
      el.setAttribute('data-capture-layout', '');
    } else {
      el.removeAttribute('data-capture-layout');
    }
  }, [search]);
  return null;
}

function App() {
  return (
    <ErrorBoundary fallbackTitle="Application Error">
      <Router>
        <CaptureLayoutSync />
        <div className="App app-layout">
          <div className="app-main">
            <Routes>
              <Route path="/" element={<RootRoute />} />
              <Route path="/analyze" element={<StockAnalysis />} />
              <Route path="/legal" element={<Legal />} />
              <Route path="/methodology" element={<Methodology />} />
              <Route path="/lime" element={<LimeExplain />} />
              <Route path="/correlation" element={<Navigate to="/analyze" replace />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </div>
          <SiteFooter />
        </div>
      </Router>
    </ErrorBoundary>
  );
}

function NotFound() {
  return (
    <div id="main-content" className="not-found" tabIndex={-1}>
      <div className="not-found__brand" aria-hidden>
        <BrandMark size={52} />
      </div>
      <h1 className="not-found__code">404</h1>
      <h2 className="not-found__title">Page not found</h2>
      <p className="not-found__text">
        The page you are looking for does not exist or has been moved.
      </p>
      <div className="not-found__links">
        <Link to="/" className="not-found__link">
          Market overview
        </Link>
        <Link to="/analyze" className="not-found__link">
          Stock analysis
        </Link>
        <Link to="/legal" className="not-found__link">
          Legal &amp; disclaimer
        </Link>
        <Link to="/methodology" className="not-found__link">
          Methodology
        </Link>
        <Link to="/lime" className="not-found__link">
          LIME explainability
        </Link>
      </div>
    </div>
  );
}

export default App;
