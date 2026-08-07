# 🚀 SecureLens Deployment Guide for Render

This guide will help you deploy SecureLens to Render.com with all your FHE functionality intact.

---

## 📋 Prerequisites

1. **GitHub Repository**: Your code must be pushed to GitHub
2. **Render Account**: Sign up at [render.com](https://render.com) (free tier available)
3. **Model Files**: Ensure your model files are committed to the repo

---

## 🎯 Deployment Steps

### Step 1: Prepare Your Repository

Make sure these files are in your repository:
- ✅ `render.yaml` (configuration file - already created)
- ✅ `requirements.txt` (dependencies - already exists)
- ✅ `app.py` (updated to support Render's PORT environment variable)
- ✅ All your model files in `cloud_server/models/`
- ✅ Your code modules: `crypto_layer/`, `cloud_server/`

### Step 2: Push to GitHub

```bash
# If not already done, initialize git and push
git add .
git commit -m "Add Render deployment configuration"
git push origin main
```

### Step 3: Deploy on Render

#### Option A: Using render.yaml (Recommended - One-Click)

1. Go to [render.com](https://render.com) and sign in
2. Click **"New +"** → **"Blueprint"**
3. Connect your GitHub repository
4. Render will automatically detect `render.yaml` and configure everything
5. Click **"Apply"** to start deployment

#### Option B: Manual Setup

1. Go to [render.com](https://render.com) and sign in
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `securelens`
   - **Region**: Oregon (US West) or closest to you
   - **Branch**: `main`
   - **Runtime**: Python 3
   - **Build Command**: 
     ```
     pip install --upgrade pip && pip install -r requirements.txt
     ```
   - **Start Command**: 
     ```
     python app.py
     ```
   - **Plan**: Free (or select paid plan for better performance)

5. Add Environment Variables (if needed):
   - Click **"Advanced"** → **"Add Environment Variable"**
   - `PYTHON_VERSION`: `3.10.12`
   - `PORT`: `7860` (optional, auto-provided by Render)

6. Click **"Create Web Service"**

---

## ⏱️ What Happens Next?

### Build Process (5-10 minutes)
Render will:
1. ✅ Clone your repository
2. ✅ Install Python 3.10
3. ✅ Install all dependencies from `requirements.txt`
4. ✅ Build your application
5. ✅ Start the Gradio app

### Monitor Deployment
- Watch the **Logs** tab in Render dashboard
- Look for: `[SecureLens] All features loaded!`
- Your app will be live at: `https://securelens-XXXX.onrender.com`

---

## 🎨 Features Available on Render

All your SecureLens features will work:
- 🔒 **TRUE FHE Classification** - Full encryption support
- ⚔️ **Attack Demo** - Adversarial robustness testing
- 📊 **Comparison** - FHE vs Traditional inference
- 🧠 **GradCAM** - Visual explainability
- 📊 **Model Evaluation** - Comprehensive metrics

---

## ⚙️ Configuration Details

### Free Tier Limits
- **RAM**: 512 MB
- **CPU**: Shared
- **Sleep**: Services on free tier sleep after 15 minutes of inactivity
- **Cold Start**: ~30 seconds to wake up
- **Build Time**: Unlimited

### Upgrade to Paid (Optional)
For better performance:
- **Starter**: $7/month - 512MB RAM, no sleep
- **Standard**: $25/month - 2GB RAM, faster CPU
- **Pro**: $85/month - 4GB RAM, dedicated CPU

---

## 🐛 Troubleshooting

### Build Fails - Dependency Issues
**Problem**: `tenseal` or `torch` installation fails

**Solution**: Render uses Ubuntu. Add a `packages` section to `render.yaml`:
```yaml
services:
  - type: web
    name: securelens
    env: python
    buildCommand: "apt-get update && apt-get install -y build-essential && pip install --upgrade pip && pip install -r requirements.txt"
```

### App Crashes - Out of Memory
**Problem**: FHE operations consume too much RAM (512MB limit on free tier)

**Solutions**:
1. Upgrade to Starter plan ($7/month) for stable service
2. Reduce model size (use smaller ResNet)
3. Add memory optimization to `app.py`

### Slow Cold Starts
**Problem**: App sleeps after inactivity, takes 30s to wake

**Solutions**:
1. Upgrade to paid plan (no sleep)
2. Use a free service like [UptimeRobot](https://uptimerobot.com/) to ping your app every 14 minutes
3. Add a health check endpoint (already configured in `render.yaml`)

### Model Files Too Large
**Problem**: GitHub has 100MB file size limit

**Solutions**:
1. Use Git LFS (Large File Storage):
   ```bash
   git lfs install
   git lfs track "*.pth"
   git lfs track "*.npy"
   git add .gitattributes
   git commit -m "Track model files with LFS"
   ```

2. Or store models externally (Hugging Face, Google Drive) and download during build

---

## 🔒 Security & Environment Variables

If you need to add secrets (API keys, tokens):

1. Go to your service in Render dashboard
2. Click **"Environment"** tab
3. Add variables:
   ```
   SECRET_KEY=your-secret-here
   API_TOKEN=your-token-here
   ```
4. Access in code: `os.environ.get('SECRET_KEY')`

---

## 📊 Monitoring & Logs

### View Logs
1. Go to your service in Render dashboard
2. Click **"Logs"** tab
3. Watch real-time logs:
   - Application startup
   - Inference requests
   - Error messages

### Metrics
- Click **"Metrics"** tab to see:
  - CPU usage
  - Memory usage
  - Request count
  - Response times

---

## 🔄 Updates & Redeployment

### Automatic Deployment
Your app auto-deploys when you push to GitHub:
```bash
git add .
git commit -m "Update model or code"
git push origin main
```
Render detects the push and redeploys automatically.

### Manual Deployment
1. Go to Render dashboard
2. Click **"Manual Deploy"** → **"Deploy latest commit"**

---

## 🌐 Custom Domain (Optional)

Add your own domain:
1. Go to **"Settings"** → **"Custom Domain"**
2. Add your domain: `securelens.yourdomain.com`
3. Update your DNS records as instructed
4. Render provides free SSL certificate

---

## 💡 Performance Tips

### 1. Optimize Model Loading
Your app already uses lazy loading - good! ✅

### 2. Add Caching
Consider caching FHE contexts to avoid regeneration:
```python
import functools

@functools.lru_cache(maxsize=1)
def get_ckks_context():
    return CKKSEngine(8192, [60, 40, 40, 60], 2**40)
```

### 3. Use Gradio Queue
Already enabled in your code - helps handle concurrent requests ✅

---

## 📞 Support

- **Render Docs**: https://render.com/docs
- **Render Community**: https://community.render.com
- **Your App Logs**: Check Render dashboard for specific errors

---

## ✅ Deployment Checklist

Before deploying, ensure:
- [ ] Code pushed to GitHub
- [ ] `render.yaml` in repository root
- [ ] `requirements.txt` includes all dependencies
- [ ] Model files committed (or accessible externally)
- [ ] `app.py` updated to use PORT environment variable
- [ ] Tested locally: `python app.py`
- [ ] Render account created

---

## 🎉 Success!

Once deployed, share your app:
- **Live URL**: `https://securelens-XXXX.onrender.com`
- **Demo**: Upload chest X-rays for FHE-powered diagnosis
- **Research**: Share with colleagues, include in papers
- **Portfolio**: Add to your GitHub README

---

## 🆚 Render vs Hugging Face Spaces

| Feature | Render | HF Spaces |
|---------|--------|-----------|
| **Free Tier RAM** | 512 MB | 2 GB |
| **Sleep** | Yes (15 min) | Yes (48h inactivity) |
| **Custom Domain** | ✅ Yes | ❌ No |
| **Build Time** | Faster | Slower |
| **Docker Support** | ✅ Yes | ✅ Yes |
| **Pricing** | $7/mo starter | $5/mo upgrade |
| **Best For** | Production apps | ML demos |

**Recommendation**: 
- Use **HF Spaces** for demos and research sharing
- Use **Render** for production, custom domains, or better control

You can deploy to **BOTH** and use them for different purposes! 🚀

---

## 📚 Next Steps

After successful deployment:
1. Test all features on live URL
2. Monitor logs for any errors
3. Share your deployment
4. Consider upgrading if you need more resources
5. Add custom domain for professional look

**Need help?** Check the troubleshooting section or Render's documentation.

**Happy Deploying! 🔐✨**
