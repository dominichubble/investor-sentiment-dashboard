import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
  Link,
  useSearchParams,
} from 'react-router-dom';
import MarketOverview from './pages/MarketOverview/MarketOverview';
import StockAnalysis from './pages/StockAnalysis';
import Legal from './pages/Legal';
import Methodology from './pages/Methodology';
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

function App() {
  return (
    <ErrorBoundary fallbackTitle="Application Error">
      <Router>
        <div className="App">
          <Routes>
            <Route path="/" element={<RootRoute />} />
            <Route path="/analyze" element={<StockAnalysis />} />
            <Route path="/legal" element={<Legal />} />
            <Route path="/methodology" element={<Methodology />} />
            <Route path="/correlation" element={<Navigate to="/analyze" replace />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </div>
      </Router>
    </ErrorBoundary>
  );
}

function NotFound() {
  return (
    <div id="main-content" className="not-found" tabIndex={-1}>
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
      </div>
    </div>
  );
}

export default App;
