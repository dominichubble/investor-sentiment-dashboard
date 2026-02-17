# Development Setup Guide

## 🚀 Quick Start Options

### Option 1: PowerShell Script (Recommended for Windows)

**Single command to start both servers:**

```powershell
.\start-dev.ps1
```

This will:
- Check dependencies
- Install frontend packages if needed
- Start backend in a new window
- Start frontend in a new window
- Show you the URLs

**URLs:**
- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

### Option 2: NPM Concurrently (Cross-platform)

**One-time setup:**

```bash
npm install
```

**Then run:**

```bash
npm run dev
```

This runs both servers in a single terminal window with nice output formatting.

**Individual commands:**
```bash
npm run dev:backend    # Backend only
npm run dev:frontend   # Frontend only
npm run install:all    # Install frontend deps
```

---

### Option 3: Docker Compose (Most Isolated)

**Build and start:**

```bash
docker-compose up --build
```

**Start (after first build):**

```bash
docker-compose up
```

**Stop:**

```bash
docker-compose down
```

**Rebuild after changes:**

```bash
docker-compose up --build
```

---

### Option 4: Manual (Two Terminals)

**Terminal 1 - Backend:**

```bash
cd backend
python -m uvicorn api.main:app --reload --host localhost --port 8000
```

**Terminal 2 - Frontend:**

```bash
cd frontend
npm install  # first time only
npm run dev
```

---

## 📋 Prerequisites

### Required:
- **Python 3.11+**
- **Node.js 18+**
- **npm 9+**

### Optional (for Docker):
- **Docker Desktop**
- **Docker Compose**

---

## 🔧 Configuration

### Environment Variables

**Backend** (`backend/.env`):
```env
# Will be used when you add database
DATABASE_URL=postgresql://sentiment_user:password@localhost:5432/investor_sentiment

# Optional
LOG_LEVEL=info
```

**Frontend** (`frontend/.env`):
```env
VITE_API_URL=http://localhost:8000/api/v1
VITE_ENV=development
```

---

## 🎯 Recommended Workflow

### Daily Development:

**Option A - Separate Windows (Best for debugging):**
```powershell
.\start-dev.ps1
```
- Easier to see logs separately
- Can restart each service independently
- Better for debugging issues

**Option B - Single Terminal (Cleaner):**
```bash
npm run dev
```
- All output in one place
- Color-coded logs
- Quick Ctrl+C stops both

---

## 🐛 Troubleshooting

### Port Already in Use

**Find and kill process:**

```powershell
# Find what's using port 8000
Get-NetTCPConnection -LocalPort 8000 | Select-Object OwningProcess
Stop-Process -Id <PID> -Force

# Or kill all python processes
Get-Process python* | Stop-Process -Force
```

### Backend Won't Start

1. Check Python version:
   ```bash
   python --version  # Should be 3.11+
   ```

2. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. Check for import errors:
   ```bash
   cd backend
   python -c "from api.main import app; print('OK')"
   ```

### Frontend Won't Start

1. Check Node version:
   ```bash
   node --version  # Should be 18+
   ```

2. Reinstall dependencies:
   ```bash
   cd frontend
   rm -rf node_modules package-lock.json
   npm install
   ```

3. Clear Vite cache:
   ```bash
   cd frontend
   rm -rf node_modules/.vite
   ```

---

## 📊 Performance Tips

### Development Mode:
- Backend auto-reloads on Python file changes
- Frontend hot-reloads on TypeScript/CSS changes
- No need to restart manually

### Production Mode:
```bash
# Frontend
cd frontend
npm run build
npm run preview

# Backend (use gunicorn in production)
cd backend
pip install gunicorn
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## 🔗 Useful URLs

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost:3000 | React Dashboard |
| Backend | http://localhost:8000 | API Root |
| API Docs | http://localhost:8000/docs | Swagger UI |
| ReDoc | http://localhost:8000/redoc | Alternative docs |
| Health | http://localhost:8000/health | Health check |

---

## 🎯 Recommended Setup

For your Windows environment, I recommend:

**Daily Development:**
```powershell
.\start-dev.ps1
```

**Quick Testing:**
```bash
npm run dev
```

**Production Deployment:**
```bash
docker-compose up -d
```

This gives you flexibility based on what you're doing!
