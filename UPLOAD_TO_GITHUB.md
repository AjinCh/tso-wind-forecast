# Upload to GitHub Instructions

## Option 1: Using GitHub Website (Easiest)

### Step 1: Create Repository on GitHub
1. Go to https://github.com/AjinCh
2. Click the **"+"** icon (top right) → **"New repository"**
3. Repository name: `tso-wind-forecast`
4. Description: `Wind speed forecasting system using LSTM neural networks with n8n automation for TenneT TSO operations`
5. Select: **Public** (or Private if you prefer)
6. **DO NOT** check "Add README" or ".gitignore" (we already have them)
7. Click **"Create repository"**

### Step 2: Push Your Code
Copy and run these commands in PowerShell:

```powershell
cd c:\Users\ajina\Documents\projects\tso-wind-forecast

# Add the remote repository
git remote add origin https://github.com/AjinCh/tso-wind-forecast.git

# Push your code
git branch -M main
git push -u origin main
```

You'll be prompted for your GitHub credentials:
- **Username:** AjinCh
- **Password:** Use a Personal Access Token (not your GitHub password)

### Step 3: Create Personal Access Token (if needed)
If you don't have a token:
1. Go to https://github.com/settings/tokens
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. Name: `tso-wind-forecast`
4. Select scopes: Check **"repo"** (full control of private repositories)
5. Click **"Generate token"**
6. **Copy the token** (you won't see it again!)
7. Use this token as your password when pushing

---

## Option 2: Using GitHub Desktop (Easiest for beginners)

1. Download GitHub Desktop: https://desktop.github.com/
2. Install and sign in with your GitHub account
3. Click **"Add"** → **"Add existing repository"**
4. Browse to: `c:\Users\ajina\Documents\projects\tso-wind-forecast`
5. Click **"Publish repository"** button
6. Name: `tso-wind-forecast`
7. Description: `Wind speed forecasting system with LSTM and n8n automation`
8. Click **"Publish repository"**

---

## Option 3: Using Git with Credential Manager

```powershell
cd c:\Users\ajina\Documents\projects\tso-wind-forecast

# Configure credential helper
git config --global credential.helper wincred

# Add remote and push
git remote add origin https://github.com/AjinCh/tso-wind-forecast.git
git branch -M main
git push -u origin main
```

Windows will open a dialog to enter your GitHub credentials.

---

## ✅ Verify Upload

After pushing, visit:
**https://github.com/AjinCh/tso-wind-forecast**

You should see all your files uploaded!

---

## 🔧 Troubleshooting

### "remote origin already exists"
```powershell
git remote remove origin
git remote add origin https://github.com/AjinCh/tso-wind-forecast.git
```

### Authentication failed
- Use a Personal Access Token (PAT) instead of your password
- Or use GitHub Desktop for easier authentication

### Want to make it private later?
Go to: https://github.com/AjinCh/tso-wind-forecast/settings
Scroll to "Danger Zone" → "Change repository visibility"
