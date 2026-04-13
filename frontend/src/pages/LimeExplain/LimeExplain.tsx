import React, { useCallback, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import Navbar from '../../components/Navbar';
import { apiService, type ExplainSentimentResponse } from '../../services/api';
import '../StockAnalysis/StockAnalysis.css';
import '../Legal/Legal.css';
import './LimeExplain.css';

const DEFAULT_TEXT =
  'Apple delivered a strong quarter and raised its dividend, but management struck a cautious tone on China demand and margin pressure from memory costs.';

const EXAMPLES: { label: string; text: string }[] = [
  { label: 'Balanced earnings', text: DEFAULT_TEXT },
  {
    label: 'Guidance / multiple',
    text: 'NVDA surged after the company raised guidance; bears argue the multiple is stretched.',
  },
  {
    label: 'Macro / Fed',
    text: 'The Fed held rates steady; markets sold off as investors parsed the dot plot for fewer cuts.',
  },
];

function isAbortError(e: unknown): boolean {
  if (axios.isCancel(e)) return true;
  if (!e || typeof e !== 'object') return false;
  const err = e as { name?: string; code?: string };
  return err.name === 'AbortError' || err.name === 'CanceledError' || err.code === 'ERR_CANCELED';
}

function badgeClass(label: string): string {
  const l = label.toLowerCase();
  if (l === 'positive') return 'lime-page__badge lime-page__badge--positive';
  if (l === 'negative') return 'lime-page__badge lime-page__badge--negative';
  return 'lime-page__badge lime-page__badge--neutral';
}

/** Diverging background: negative weights cool, positive weights warm. */
function tokenStyle(weight: number, maxAbs: number): React.CSSProperties {
  if (!maxAbs || !Number.isFinite(weight)) {
    return { background: 'rgba(148, 163, 184, 0.2)' };
  }
  const t = Math.max(-1, Math.min(1, weight / maxAbs));
  if (t >= 0) {
    const a = 0.12 + 0.38 * t;
    return { background: `rgba(220, 38, 38, ${a})`, borderColor: 'rgba(220, 38, 38, 0.25)' };
  }
  const a = 0.12 + 0.38 * -t;
  return { background: `rgba(37, 99, 235, ${a})`, borderColor: 'rgba(37, 99, 235, 0.22)' };
}

function formatDetail(err: unknown): string {
  if (typeof err === 'object' && err !== null && 'response' in err) {
    const r = (err as { response?: { status?: number; data?: { detail?: unknown } } }).response;
    const d = r?.data?.detail;
    if (typeof d === 'string') return d;
    if (Array.isArray(d)) return d.map((x) => JSON.stringify(x)).join('; ');
  }
  if (err instanceof Error) return err.message;
  return 'Request failed. Check the API is running and LIME is not disabled (lean mode).';
}

export default function LimeExplain() {
  const [text, setText] = useState(DEFAULT_TEXT);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ExplainSentimentResponse | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const maxAbsWeight = useMemo(() => {
    if (!result?.tokens?.length) return 0;
    return result.tokens.reduce((m, t) => Math.max(m, Math.abs(t.weight)), 0);
  }, [result]);

  const runExplain = useCallback(async () => {
    const trimmed = text.trim();
    if (!trimmed) {
      setError('Enter some financial text to explain.');
      setResult(null);
      return;
    }
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await apiService.explainSentiment(
        trimmed,
        { num_features: 14, num_samples: 500 },
        ac.signal,
      );
      setResult(data);
    } catch (e) {
      if (isAbortError(e)) return;
      setError(formatDetail(e));
    } finally {
      setLoading(false);
    }
  }, [text]);

  return (
    <div id="main-content" className="legal-page stock-analysis" tabIndex={-1}>
      <Navbar
        title="LIME explainability"
        subtitle="Local token-level explanations for FinBERT sentiment on any pasted financial sentence."
        siteNav
      />

      <article className="legal-page__article lime-page__article">
        <p className="lime-page__lead">
          This page calls the production <strong>POST /api/v1/sentiment/explain</strong> endpoint (LIME over
          FinBERT). Explanations are <strong>local linear approximations</strong> and can misrepresent negation
          or long-range structure; use them to inspect model focus, not as ground truth.
        </p>

        <div className="lime-page__controls">
          <label htmlFor="lime-text" className="lime-page__label">
            Text to explain
          </label>
          <textarea
            id="lime-text"
            className="lime-page__textarea"
            value={text}
            onChange={(e) => setText(e.target.value)}
            spellCheck
            maxLength={8000}
            aria-describedby="lime-text-hint"
          />
          <p id="lime-text-hint" className="lime-page__hint">
            Tip: paste headlines, Reddit lines, or tweets. Very long posts may take up to two minutes on CPU.
          </p>
        </div>

        <div className="lime-page__actions">
          <button
            type="button"
            className="lime-page__btn"
            onClick={() => void runExplain()}
            disabled={loading}
            aria-busy={loading}
          >
            {loading ? 'Running LIME…' : 'Run LIME explanation'}
          </button>
          <div className="lime-page__examples" role="group" aria-label="Example texts">
            <span className="lime-page__hint">Examples:</span>
            {EXAMPLES.map((ex) => (
              <button
                key={ex.label}
                type="button"
                className="lime-page__btn lime-page__btn-secondary"
                onClick={() => setText(ex.text)}
                disabled={loading}
              >
                {ex.label}
              </button>
            ))}
          </div>
        </div>

        {error && (
          <div className="lime-page__error" role="alert">
            {error}
          </div>
        )}

        {result && (
          <section className="lime-page__result" aria-live="polite">
            <div className="lime-page__result-head">
              <span className={badgeClass(result.prediction.label)}>{result.prediction.label}</span>
              <span>
                confidence <strong>{(result.prediction.score * 100).toFixed(1)}%</strong>
              </span>
            </div>

            {result.prediction.scores && (
              <div className="lime-page__scores">
                <div className="lime-page__score">
                  Positive
                  <strong>{(result.prediction.scores.positive * 100).toFixed(1)}%</strong>
                </div>
                <div className="lime-page__score">
                  Neutral
                  <strong>{(result.prediction.scores.neutral * 100).toFixed(1)}%</strong>
                </div>
                <div className="lime-page__score">
                  Negative
                  <strong>{(result.prediction.scores.negative * 100).toFixed(1)}%</strong>
                </div>
              </div>
            )}

            <h3 className="lime-page__heatmap-title">Token contributions (LIME)</h3>
            <div className="lime-page__heatmap" lang="en">
              {result.tokens.map((t, i) => (
                <span
                  key={`${i}-${t.token}`}
                  className="lime-page__token"
                  style={tokenStyle(t.weight, maxAbsWeight)}
                  title={`weight: ${t.weight.toFixed(4)}`}
                >
                  {t.token === '[CLS]' || t.token === '[SEP]' ? (
                    <span className="lime-page__hint">{t.token}</span>
                  ) : (
                    t.token.replace(/\s+/g, '\u00a0')
                  )}
                </span>
              ))}
            </div>

            <p className="lime-page__meta">
              {result.metadata.method} · {result.metadata.num_features} features ·{' '}
              {result.metadata.num_samples} perturbations · {result.metadata.processing_time_ms} ms ·{' '}
              {result.metadata.timestamp}
            </p>
          </section>
        )}

        <p className="legal-page__back" style={{ marginTop: '2rem' }}>
          <Link to="/analyze" className="legal-page__back-link">
            Stock analysis (correlation &amp; time series)
          </Link>
          {' · '}
          <Link to="/methodology" className="legal-page__back-link">
            Methodology
          </Link>
          {' · '}
          <Link to="/" className="legal-page__back-link">
            Market overview
          </Link>
        </p>
      </article>
    </div>
  );
}
