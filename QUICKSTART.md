# 🚀 Quick Start Guide

## Three ways to run both servers

### 1️⃣ PowerShell Script (Easiest for Windows)

```powershell
.\start-dev.ps1
```

**What it does:**
- ✅ Checks all dependencies
- ✅ Installs frontend packages if needed
- ✅ Opens backend in new window
- ✅ Opens frontend in new window
- ✅ Shows URLs for both

**Best for:** Daily development, easier debugging

---

### 2️⃣ NPM Script (Single Terminal)

```bash
npm run dev
```

**What it does:**
- ✅ Runs both servers in one terminal
- ✅ Color-coded output (backend/frontend)
- ✅ Ctrl+C stops both

**First time only:**
```bash
npm install              # Install concurrently
cd frontend && npm install  # Install frontend deps
```

**Best for:** Quick testing, cleaner output

---

### 3️⃣ Docker Compose (Most Isolated)

```bash
docker-compose up
```

**What it does:**
- ✅ Containerized environment
- ✅ Consistent across machines
- ✅ Easy to add databases later

**First time:**
```bash
docker-compose up --build
```

**Best for:** Production-like environment

---

## 📱 Access Your App

After starting with any method:

- **Dashboard**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## 🛑 Stopping Servers

### PowerShell Script:
Press `Ctrl+C` in each window

### NPM Script:
Press `Ctrl+C` once (stops both)

### Docker:
```bash
docker-compose down
```

---

## ⚡ Pro Tips

1. **First time setup:**
   ```bash
   cd frontend && npm install
   cd ../backend && pip install -r requirements.txt
   ```

2. **Check if running:**
   ```bash
   curl http://localhost:8000/health
   ```

3. **Kill stuck processes:**
   ```powershell
   Get-Process python* | Stop-Process -Force
   Get-Process node* | Stop-Process -Force
   ```

---

## 🎯 My Recommendation

**For you (Windows developer):**

Use **`.\start-dev.ps1`** — it's the most Windows-friendly, gives you separate terminals for easier debugging, and handles setup automatically.

Just run:
```powershell
.\start-dev.ps1
```

Done! Both servers start and you can see logs separately.
