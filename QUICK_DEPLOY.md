# ⚡ Quick Deployment Reference

## 🎯 Render (5 Minutes)

### Option A: One Command
```powershell
.\deploy_to_render.ps1
```
Then go to [render.com](https://render.com) → New Blueprint → Connect repo

### Option B: Manual
```bash
git add .
git commit -m "Deploy to Render"
git push origin main
```
Then: render.com → New Web Service → Connect repo

---

## 📋 Pre-Deployment Checklist

- [ ] Code pushed to GitHub
- [ ] `render.yaml` exists (✅ created)
- [ ] `requirements.txt` exists (✅ exists)
- [ ] Model files included or accessible
- [ ] `app.py` updated (✅ updated)

---

## 🔗 Quick Links

- **Render Dashboard**: https://dashboard.render.com
- **GitHub Repo**: Push your code here first
- **Deployment Guide**: See `RENDER_DEPLOYMENT.md`
- **Platform Comparison**: See `DEPLOYMENT_OPTIONS.md`

---

## ⚙️ Configuration Files Created

1. ✅ `render.yaml` - Render configuration (auto-detected)
2. ✅ `app.py` - Updated with PORT support
3. ✅ `.gitattributes` - Git LFS for large files
4. ✅ `deploy_to_render.ps1` - Deployment helper

---

## 🐛 Common Issues

### "Port already in use"
Already fixed! `app.py` now uses environment PORT variable.

### "Module not found"
Check `requirements.txt` includes all dependencies.

### "Out of memory"
Upgrade to Render Starter plan ($7/mo) for 512MB guaranteed RAM.

### "Build timeout"
Normal for first build. Subsequent builds are faster (cached).

---

## 💡 After Deployment

Your app will be live at:
```
https://securelens-XXXX.onrender.com
```

Test all features:
- [ ] FHE Classification works
- [ ] Attack Demo works
- [ ] Comparison works
- [ ] GradCAM works
- [ ] Model Evaluation works

---

## 🎉 Success Criteria

When you see this in logs:
```
[SecureLens] All features loaded!
```

Your deployment is successful! 🚀

---

## 📞 Need Help?

- Full guide: `RENDER_DEPLOYMENT.md`
- Platform comparison: `DEPLOYMENT_OPTIONS.md`
- Render docs: https://render.com/docs
