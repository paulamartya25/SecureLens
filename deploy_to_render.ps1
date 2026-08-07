# SecureLens - Render Deployment Script (PowerShell)
# This script helps you prepare and deploy SecureLens to Render

Write-Host "SecureLens - Render Deployment Helper" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is installed
Write-Host "[1/5] Checking Git installation..." -ForegroundColor Yellow
try {
    $gitVersion = git --version
    Write-Host "✅ Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Git not found! Please install Git first." -ForegroundColor Red
    Write-Host "Download from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

# Check if we're in a git repository
Write-Host "`n[2/5] Checking Git repository..." -ForegroundColor Yellow
if (Test-Path .git) {
    Write-Host "✅ Git repository found" -ForegroundColor Green
} else {
    Write-Host "⚠️  Not a git repository. Initializing..." -ForegroundColor Yellow
    git init
    Write-Host "✅ Git repository initialized" -ForegroundColor Green
}

# Check for required files
Write-Host "`n[3/5] Checking required files..." -ForegroundColor Yellow
$requiredFiles = @(
    "render.yaml",
    "requirements.txt",
    "app.py",
    "app_gradio_enhanced.py"
)

$allFilesExist = $true
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  ✅ $file" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $file (missing!)" -ForegroundColor Red
        $allFilesExist = $false
    }
}

if (-not $allFilesExist) {
    Write-Host "`n❌ Some required files are missing!" -ForegroundColor Red
    exit 1
}

# Check model files
Write-Host "`n[4/5] Checking model files..." -ForegroundColor Yellow
$modelPath = "cloud_server\models"
if (Test-Path $modelPath) {
    $modelFiles = Get-ChildItem $modelPath -File
    Write-Host "  ✅ Found $($modelFiles.Count) model files in $modelPath" -ForegroundColor Green
    
    # Check for large files
    $largeFiles = $modelFiles | Where-Object { $_.Length -gt 100MB }
    if ($largeFiles.Count -gt 0) {
        Write-Host ""
        Write-Host "  WARNING: Large model files detected (>100MB):" -ForegroundColor Yellow
        foreach ($file in $largeFiles) {
            $sizeMB = [math]::Round($file.Length / 1MB, 2)
            $sizeText = "$sizeMB MB"
            Write-Host "    - $($file.Name) ($sizeText)" -ForegroundColor Yellow
        }
        Write-Host ""
        Write-Host "  Consider using Git LFS for files >100MB:" -ForegroundColor Cyan
        Write-Host "     git lfs install" -ForegroundColor Gray
        Write-Host "     git lfs track '*.pth'" -ForegroundColor Gray
        Write-Host "     git lfs track '*.npy'" -ForegroundColor Gray
    }
} else {
    Write-Host "  ⚠️  Model directory not found: $modelPath" -ForegroundColor Yellow
}

# Git status
Write-Host ""
Write-Host "[5/5] Checking Git status..." -ForegroundColor Yellow
$status = git status --porcelain
if ($status) {
    Write-Host "  You have uncommitted changes" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Would you like to commit and push now? (Y/N)" -ForegroundColor Cyan
    $response = Read-Host
    
    if ($response -eq 'Y' -or $response -eq 'y') {
        Write-Host ""
        Write-Host "Staging files..." -ForegroundColor Yellow
        git add .
        
        Write-Host "Enter commit message (or press Enter for default):" -ForegroundColor Cyan
        $commitMsg = Read-Host
        if ([string]::IsNullOrWhiteSpace($commitMsg)) {
            $commitMsg = "Add Render deployment configuration"
        }
        
        Write-Host "Committing changes..." -ForegroundColor Yellow
        git commit -m $commitMsg
        
        Write-Host ""
        Write-Host "Checking remote repository..." -ForegroundColor Yellow
        $remotes = git remote -v
        if ($remotes) {
            Write-Host "Pushing to remote..." -ForegroundColor Yellow
            try {
                git push
                Write-Host "Successfully pushed to remote!" -ForegroundColor Green
            } catch {
                Write-Host "Push failed. You may need to set up remote first:" -ForegroundColor Red
                Write-Host "   git remote add origin <your-repo-url>" -ForegroundColor Gray
                Write-Host "   git push -u origin main" -ForegroundColor Gray
            }
        } else {
            Write-Host "No remote repository configured." -ForegroundColor Yellow
            Write-Host "Please add your GitHub repository:" -ForegroundColor Cyan
            Write-Host "  git remote add origin <your-repo-url>" -ForegroundColor Gray
            Write-Host "  git push -u origin main" -ForegroundColor Gray
        }
    }
} else {
    Write-Host "  Working directory clean - ready to deploy!" -ForegroundColor Green
}

# Final instructions
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Preparation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "1. Go to https://render.com and sign in" -ForegroundColor White
Write-Host "2. Click 'New +' → 'Blueprint' (or 'Web Service')" -ForegroundColor White
Write-Host "3. Connect your GitHub repository" -ForegroundColor White
Write-Host "4. Render will auto-detect render.yaml" -ForegroundColor White
Write-Host "5. Click 'Apply' to deploy!" -ForegroundColor White

Write-Host "`n📚 For detailed instructions, see:" -ForegroundColor Cyan
Write-Host "   RENDER_DEPLOYMENT.md" -ForegroundColor Gray

Write-Host "`n✨ Your deployment is ready! Good luck! 🚀" -ForegroundColor Green
