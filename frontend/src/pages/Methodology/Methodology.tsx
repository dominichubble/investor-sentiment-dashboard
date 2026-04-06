import React from 'react';
import { Link } from 'react-router-dom';
import Navbar from '../../components/Navbar';
import '../StockAnalysis/StockAnalysis.css';
import '../Legal/Legal.css';
import './Methodology.css';

/**
 * Examiner-oriented map from dissertation claims to the live artefact.
 * British English; limitations foregrounded for academic credibility.
 */
export default function Methodology() {
  return (
    <div id="main-content" className="legal-page stock-analysis" tabIndex={-1}>
      <Navbar
        title="Methodology &amp; evaluation"
        subtitle="How this artefact supports a dissertation on sentiment analysis for financial assets — and what it does not claim."
        siteNav
      />

      <article className="legal-page__article methodology-page__article">
        <p className="methodology-page__lead">
          <strong>For examiners:</strong> this page ties the written dissertation to the running system. Use it
          during the demo or viva to jump from a <em>claim</em> (pipeline, metrics, limitations) to the
          corresponding <em>screen or file</em> in the repository.
        </p>

        <h2 className="methodology-page__h2" id="contribution">
          Contribution (what the artefact is for)
        </h2>
        <p>
          The project implements an <strong>end-to-end pipeline</strong> from heterogeneous text (social and news)
          to stored labels, aggregates, and <strong>exploratory</strong> alignment with price returns — exposed
          through a FastAPI backend and this React dashboard. The scholarly focus is typically: (1) domain-suited
          sentiment classification, (2) transparent evaluation against a baseline, (3) interpretability where
          appropriate, and (4) honest scoping of what correlation and causality tests can and cannot show.
        </p>

        <h2 className="methodology-page__h2" id="pipeline">
          Data and processing pipeline
        </h2>
        <ol className="methodology-page__list methodology-page__list--pipeline">
          <li>
            <strong>Ingestion</strong> — Text is collected via configured sources (e.g. Reddit, news APIs, historical
            social datasets) and stored as stock-associated records.
          </li>
          <li>
            <strong>Classification</strong> — FinBERT (or equivalent financial sentiment model) assigns positive /
            neutral / negative labels; a keyword baseline supports comparative evaluation in the dissertation.
          </li>
          <li>
            <strong>Aggregation</strong> — Daily (and optionally smoothed) net sentiment and mention counts feed
            descriptive views and correlation inputs.
          </li>
          <li>
            <strong>Price alignment</strong> — Returns are aligned to sentiment by calendar/trading-day rules you
            specify in the UI (same day vs next day; optional market adjustment).
          </li>
          <li>
            <strong>Presentation</strong> — Charts, summary statistics, optional LLM-generated narrative from the
            same window (clearly labelled as model output, not ground truth).
          </li>
        </ol>
        <p className="methodology-page__note">
          Code layout mirrors this story: see <code className="legal-page__code">backend/app/pipelines</code>,{' '}
          <code className="legal-page__code">models</code>, <code className="legal-page__code">analysis</code>, and{' '}
          <code className="legal-page__code">evaluation</code> in the repository README.
        </p>

        <h2 className="methodology-page__h2" id="mapping">
          Where to find each evaluation theme
        </h2>
        <div className="methodology-page__map-wrap">
          <table className="methodology-page__map">
            <thead>
              <tr>
                <th scope="col">Dissertation theme</th>
                <th scope="col">In the artefact</th>
                <th scope="col">In the repo</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Classifier accuracy vs baseline</td>
                <td>Discussed in README; not a live chart</td>
                <td>
                  <code className="legal-page__code">backend/data/evaluation/benchmark_results.json</code>, benchmark
                  module under <code className="legal-page__code">app/evaluation</code>
                </td>
              </tr>
              <tr>
                <td>Explainability (LIME)</td>
                <td>Supporting evidence for interpretability chapter</td>
                <td>
                  <code className="legal-page__code">data/processed/explanations/</code>,{' '}
                  <code className="legal-page__code">backend/scripts/generate_lime_examples.py</code>
                </td>
              </tr>
              <tr>
                <td>Correlational analysis</td>
                <td>
                  <Link to="/analyze">Stock analysis</Link> — Pearson/Spearman, lag grid, rolling correlation,
                  optional Granger and time holdout
                </td>
                <td>
                  <code className="legal-page__code">backend/app/analysis/correlation.py</code>, correlation router
                </td>
              </tr>
              <tr>
                <td>Data coverage and noise</td>
                <td>Data quality panel on stock results; market KPIs on overview</td>
                <td>Statistics and quality endpoints (see API routers)</td>
              </tr>
              <tr>
                <td>Cross-source disagreement</td>
                <td>
                  <Link to="/">Market overview</Link> — &ldquo;Cross-source disagreement over time&rdquo; (all
                  sources only)
                </td>
                <td>
                  <code className="legal-page__code">app/analysis/source_disagreement.py</code>,{' '}
                  <code className="legal-page__code">statistics_service._build_source_disagreement_trend</code>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <h2 className="methodology-page__h2" id="statistics">
          Statistical methods (plain terms + scope)
        </h2>
        <ul className="methodology-page__list">
          <li>
            <strong>Pearson / Spearman</strong> — Measure linear (Pearson) or rank-based (Spearman) association
            between aggregated sentiment and returns over the chosen window. They do <strong>not</strong> prove
            causation or tradable edge.
          </li>
          <li>
            <strong>Lag search</strong> — Explores simple lead/lag structure; significance flags are conventional
            and should be interpreted alongside multiple comparisons and sample size (discuss in the thesis).
          </li>
          <li>
            <strong>Granger-style tests</strong> — Ask whether lagged values of one series help predict another
            within a linear VAR-style framing — a <strong>statistical</strong> notion of &ldquo;predictive
            content&rdquo;, not a trading strategy backtest.
          </li>
          <li>
            <strong>Holdout / time split</strong> — A sanity check that a fitted association is not only an artefact
            of the earliest part of the sample; still not out-of-sample validation of a strategy.
          </li>
        </ul>

        <h2 className="methodology-page__h2" id="limitations">
          Limitations you should cite in the dissertation
        </h2>
        <ul className="methodology-page__list">
          <li>
            <strong>Selection and survivorship</strong> — Buzz and liquidity correlate with which tickers appear
            often in text and price data.
          </li>
          <li>
            <strong>Confounding</strong> — News, macro moves, and sector factors drive both mood and prices; market
            adjustment reduces but does not remove this.
          </li>
          <li>
            <strong>Non-stationarity</strong> — Relationships drift; rolling views illustrate instability rather than
            a single universal law.
          </li>
          <li>
            <strong>Labelling error</strong> — FinBERT can misfire on sarcasm, context, and entity linkage; quality
            flags are heuristics, not gold-standard audits.
          </li>
          <li>
            <strong>Multiple testing</strong> — Many lags and metrics increase the chance of spurious significance
            unless you apply corrections or pre-register hypotheses (state your choice in the thesis).
          </li>
        </ul>

        <h2 className="methodology-page__h2" id="reproducibility">
          Reproducibility
        </h2>
        <p>
          Follow the root <strong>README</strong> (Quick Start): Python virtual environment,{' '}
          <code className="legal-page__code">.env</code> from <code className="legal-page__code">.env.example</code>,
          database URL, then <code className="legal-page__code">pytest</code> for regression tests. The lean API mode
          documents which endpoints require full ML stacks — state which configuration you used for examiner demos.
        </p>

        <nav className="methodology-page__footer-nav" aria-label="Related pages">
          <Link to="/">Market overview</Link>
          <Link to="/analyze">Stock analysis</Link>
          <Link to="/legal">Legal &amp; disclaimer</Link>
        </nav>
      </article>
    </div>
  );
}
