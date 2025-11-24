# Deployment Guide

## ⚠️ Important: Netlify is NOT suitable for Flask apps

Netlify is designed for **static sites** and **serverless functions**, not long-running Flask applications. Your Flask app needs:
- Persistent server process
- Database connections
- In-memory state (chat sessions)
- File system access (RAG index, ChromaDB)

## ✅ Recommended Platforms for Flask Apps

### Option 1: **Render** (Easiest - Recommended)
Your code already mentions Render! This is the best option.

**Steps:**
1. Create account at [render.com](https://render.com)
2. Connect your GitHub repository
3. Create a new **Web Service**
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Environment**: Python 3
5. Add environment variables in Render dashboard:
   - `GOOGLE_API_KEY`
   - `DB_HOST`
   - `DB_USER`
   - `DB_PW`
   - `DB_NAME`
   - `DB_PORT`
6. Deploy!

**Pros:**
- Free tier available
- Easy setup
- Automatic HTTPS
- Environment variable management
- Your code already references it

**Cons:**
- Free tier spins down after inactivity (15 min)
- Limited resources on free tier

---

### Option 2: **Railway** (Great for development)
Modern platform, very easy to use.

**Steps:**
1. Create account at [railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway auto-detects Flask and sets up
5. Add environment variables in dashboard
6. Deploy!

**Pros:**
- Very easy setup
- Good free tier ($5 credit/month)
- Fast deployments
- Great developer experience

**Cons:**
- Free tier has usage limits

---

### Option 3: **Fly.io** (Good performance)
Great for apps that need to stay warm.

**Steps:**
1. Install Fly CLI: `curl -L https://fly.io/install.sh | sh`
2. Login: `fly auth login`
3. Initialize: `fly launch` (in Final/ directory)
4. Add environment variables: `fly secrets set KEY=value`
5. Deploy: `fly deploy`

**Pros:**
- Apps stay warm (no cold starts)
- Good free tier
- Global edge network
- Fast

**Cons:**
- Requires CLI setup
- More configuration needed

---

### Option 4: **Heroku** (Classic, but paid)
**Note:** Heroku removed free tier, but still good option.

**Steps:**
1. Create account at [heroku.com](https://heroku.com)
2. Install Heroku CLI
3. Login: `heroku login`
4. Create app: `heroku create your-app-name`
5. Add environment variables: `heroku config:set KEY=value`
6. Deploy: `git push heroku main`

**Pros:**
- Very mature platform
- Excellent documentation
- Add-ons ecosystem

**Cons:**
- No free tier anymore
- More expensive

---

## 📋 Pre-Deployment Checklist

Before deploying, ensure you have:

1. **Procfile** (for Render/Railway/Heroku):
   ```
   web: gunicorn app:app
   ```

2. **Runtime file** (optional, for Python version):
   ```
   python-3.11.0
   ```
   Or create `runtime.txt` with: `python-3.11.0`

3. **Environment variables** ready:
   - `GOOGLE_API_KEY`
   - `DB_HOST`, `DB_USER`, `DB_PW`, `DB_NAME`, `DB_PORT`
   - `PORT` (usually auto-set by platform)

4. **Database accessible** from internet (if using remote MySQL)

5. **RAG index built** (ChromDB files should be in repo or built on deploy)

---

## 🚀 Quick Start: Render (Recommended)

1. **Create Procfile**:
   ```
   web: gunicorn app:app
   ```

2. **Push to GitHub** (if not already)

3. **Go to Render Dashboard**:
   - New → Web Service
   - Connect GitHub repo
   - Select your repo

4. **Configure**:
   - Name: `your-app-name`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Plan: Free (or paid for better performance)

5. **Add Environment Variables**:
   - Go to Environment tab
   - Add all required variables

6. **Deploy!**

---

## 🔧 If You MUST Use Netlify (Not Recommended)

Netlify would require:
1. Converting Flask routes to Netlify Functions (serverless)
2. Refactoring to stateless architecture
3. Moving chat sessions to database/external storage
4. Significant code changes

**This is NOT recommended** - use Render, Railway, or Fly.io instead.

---

## 📝 Additional Notes

- **ChromaDB**: Make sure `chroma_db/` directory is either:
  - Committed to repo (if small)
  - Built on first deploy via build script
  - Stored externally and downloaded on startup

- **Static files**: Your `static/` and `templates/` folders should work as-is

- **Database**: Ensure your MySQL database allows connections from the hosting platform's IP addresses

- **CORS**: Already configured in your app, should work fine

---

## 🆘 Troubleshooting

**App won't start:**
- Check logs in platform dashboard
- Verify environment variables are set
- Ensure `gunicorn` is in requirements.txt

**Database connection fails:**
- Check database allows external connections
- Verify firewall rules
- Test connection from platform's IP

**RAG not working:**
- Ensure ChromaDB files are present
- Check build script runs `build_rag_index.py` if needed
- Verify file paths are correct

