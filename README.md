# Investor Sentiment Dashboard  
### BSc Computer Science – Final Year Project  
**Author:** Dominic Hubble (F319895)  
**Supervisor:** Professor Stephen Lynch  
**Department:** Computer Science, Loughborough University  
**Academic Year:** 2025–2026  

---

## 🧭 Project Overview  

The **Investor Sentiment Dashboard** is an interactive analytics system that uses Natural Language Processing (NLP) and Machine Learning (ML) to analyse public sentiment surrounding financial assets, including **ETFs**, **cryptocurrencies**, and **stocks**.  

The goal of this project is to explore how social and news media sentiment correlates with market trends and to present those findings transparently through an explainable AI interface.  

---

## 🧠 Objectives  

- Aggregate sentiment data from **Reddit**, **X (Twitter)**, and **financial news APIs**.  
- Apply **FinBERT** for financial-domain sentiment classification (positive / negative / neutral).  
- Integrate **Explainable AI** tools such as **SHAP** and **LIME** to interpret model predictions.  
- Build an interactive, web-based dashboard using **React** (frontend) and **FastAPI** (backend).  
- Evaluate accuracy, interpretability, and usability of the system.  

---

## 📚 Documentation

### Quick Start
- [Backend Setup](backend/README.md) - Python backend installation and usage
- [Notebooks](notebooks/README.md) - Jupyter notebooks for exploration

### Detailed Guides
- [Data Pipeline](docs/data-pipeline.md) - Data collection and processing workflow
- [Preprocessing Guide](docs/preprocessing-guide.md) - Text preprocessing configurations
- [FinBERT Model](docs/finbert-model.md) - Sentiment analysis API reference
- [FinBERT Implementation](docs/finbert-implementation.md) - Architecture and technical details

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+ (for frontend)
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/dominichubble/investor-sentiment-dashboard.git
cd investor-sentiment-dashboard

# Set up Python backend
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
pip install -r backend/requirements.txt

# Initialize FinBERT model
cd backend
python -m app.models.init_finbert
```

See [Backend README](backend/README.md) for detailed setup instructions.

---

## 📂 Project Structure

```
investor-sentiment-dashboard/
├── backend/                  # Python backend
│   ├── app/
│   │   ├── pipelines/       # Data ingestion scripts
│   │   ├── preprocessing/   # Text processing modules
│   │   └── models/          # ML models (FinBERT)
│   └── tests/               # Unit tests
├── frontend/                # React dashboard (coming soon)
├── notebooks/               # Jupyter notebooks for exploration
├── docs/                    # Documentation
│   ├── data-pipeline.md
│   ├── preprocessing-guide.md
│   ├── finbert-model.md
│   └── finbert-implementation.md
└── data/                    # Data storage
    ├── raw/                 # Raw ingested data
    └── processed/           # Preprocessed data
```

---

## 🛠️ Technology Stack

- **Backend:** Python 3.11, FastAPI
- **ML/NLP:** FinBERT (Transformers), PyTorch, NLTK
- **Data Sources:** Reddit API (PRAW), Twitter API (Tweepy), NewsAPI
- **Frontend:** React, D3.js (planned)
- **Testing:** pytest
- **Version Control:** Git, GitHub

---

## 📊 Current Status

### ✅ Completed
- Data ingestion pipelines (Reddit, News API)
- Text preprocessing optimized for FinBERT
- FinBERT sentiment analysis implementation
- Model caching and GPU/CPU fallback
- Comprehensive test suite
- Documentation and guides

### 🚧 In Progress
- Frontend dashboard development
- API endpoint implementation
- Explainable AI integration (SHAP, LIME)

### 📋 Planned
- Real-time sentiment monitoring
- Historical trend analysis
- Multi-asset comparison dashboard
- Performance evaluation and benchmarking

---

## 📄 License

This project is part of academic research at Loughborough University.  