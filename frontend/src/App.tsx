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
    <div style={{
      display: 'flex',
      flexDirection: 'column' as const,
      alignItems: 'center',
      justifyContent: 'center',
      height: '80vh',
      color: '#666',
      textAlign: 'center' as const,
      fontFamily: "'Roboto', sans-serif",
    }}>
      <h1 style={{ fontSize: '72px', marginBottom: '8px', color: '#5c7cfa' }}>404</h1>
      <h2 style={{ marginBottom: '16px', color: '#333' }}>Page Not Found</h2>
      <p style={{ marginBottom: '24px' }}>The page you are looking for does not exist.</p>
      <Link
        to="/"
        style={{
          padding: '10px 24px',
          borderRadius: '8px',
          background: '#5c7cfa',
          color: 'white',
          textDecoration: 'none',
          fontWeight: 500,
        }}
      >
        Go to analysis
      </Link>
    </div>
  );
}

export default App;
