import { BrowserRouter as Router, Routes, Route, Navigate, Link } from 'react-router-dom';
import StockAnalysis from './pages/StockAnalysis';
import { ErrorBoundary } from './components/ErrorBoundary';
import './App.css';

function App() {
  return (
    <ErrorBoundary fallbackTitle="Application Error">
      <Router>
        <div className="App">
          <Routes>
            <Route path="/" element={<StockAnalysis />} />
            <Route path="/correlation" element={<Navigate to="/" replace />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </div>
      </Router>
    </ErrorBoundary>
  );
}

function NotFound() {
  return (
    <div className="not-found">
      <h1 className="not-found__code">404</h1>
      <h2 className="not-found__title">Page not found</h2>
      <p className="not-found__text">
        The page you are looking for does not exist or has been moved.
      </p>
      <Link to="/" className="not-found__link">
        Back to analysis
      </Link>
    </div>
  );
}

export default App;
