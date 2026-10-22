# DocuSense AI — Deployment Guide

This guide provides step-by-step instructions for deploying DocuSense AI using the root `Dockerfile` to free cloud container hosts:
1. **Hugging Face Spaces** (Docker SDK — Recommended)
2. **Render** (Web Service Docker)

---

## Platform Verification & Specifications

The deployment configuration was verified against the latest official platform documentation:

| Specification | Hugging Face Spaces (Docker SDK) | Render (Web Service Docker) |
|---|---|---|
| **Port Handling** | Default expected port is `7860`. Container must bind to `0.0.0.0:7860`. | Injects `PORT` environment variable (defaults to `10000`). Container must bind to `0.0.0.0:$PORT`. |
| **User Privileges** | **Strict requirement**: Must run as a non-root user with **UID 1000**. Running as root causes permission errors. | Non-root users supported and encouraged. |
| **Free Tier Resources** | **2 vCPU, 16 GB RAM** (Community tier). | **~0.1 vCPU, 512 MB RAM**. |
| **Inactivity Policy** | Remains active while traffic exists; restarts quickly. | Spins down after **15 minutes of inactivity**; first request takes 30–50s cold start. |
| **Filesystem** | **Ephemeral** disk. Resets upon container restart. | **Ephemeral** disk (persistent disks require paid plan). |
| **Recommendation** | **Best for DocuSense AI**: 16 GB RAM comfortably handles high-resolution OCR scans. | Viable, but 512 MB RAM can be constrained during concurrent 300 DPI PDF rasterization. |

> [!NOTE]
> The root `Dockerfile` uses the unified command:
> `CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]`
> This dynamically uses Render's injected `$PORT` (`10000`) or defaults to `7860` for Hugging Face Spaces and local runs.

---

## Method 1: Hugging Face Spaces (Recommended)

Hugging Face Spaces provides a free 16 GB RAM environment with native Docker SDK support.

### Step 1: Create a New Space
1. Log into [huggingface.co](https://huggingface.co).
2. Click your profile icon and select **New Space** (or navigate to `https://huggingface.co/new-space`).
3. Set your Space details:
   - **Space name**: `docusense-ai`
   - **License**: `mit` or `apache-2.0`
   - **Space SDK**: Select **Docker** -> **Blank**.
   - **Space hardware**: Select **Free (2 vCPU · 16 GB RAM)**.
   - **Visibility**: Public or Private.
4. Click **Create Space**.

### Step 2: Configure Space Metadata
Hugging Face Spaces reads configuration from the YAML frontmatter at the top of `README.md`.
Ensure the repository's root `README.md` begins with this block:

```yaml
---
title: DocuSense AI
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---
```

### Step 3: Push Code to the Space
Clone your new Space repository and push your project files (or add Hugging Face as a git remote):

```bash
# Add Hugging Face Space as a remote
git remote add space https://huggingface.co/spaces/<YOUR-USERNAME>/docusense-ai

# Push to Hugging Face
git push space main
```

*(Alternatively, use the Hugging Face web UI to upload the files directly).*

### Step 4: Verify Build and Access
1. Hugging Face Spaces will automatically detect the `Dockerfile`, build the image, install Tesseract OCR and dependencies, and launch Uvicorn.
2. Monitor the build logs on the **Building** tab.
3. Once the status shows **Running**, access your web interface directly in your browser:
   `https://huggingface.co/spaces/<YOUR-USERNAME>/docusense-ai`

---

## Method 2: Render (Web Service Docker)

Render can build and run Docker containers directly from a connected GitHub or GitLab repository.

### Step 1: Push Project to GitHub
Ensure your repository is pushed to GitHub or GitLab with `Dockerfile` and `.dockerignore` at the root.

### Step 2: Create a Web Service on Render
1. Log into [dashboard.render.com](https://dashboard.render.com).
2. Click **New +** -> **Web Service**.
3. Select **Build and deploy from a Git repository** and connect your repository.
4. Fill in the service configuration:
   - **Name**: `docusense-ai`
   - **Region**: Choose the closest region (e.g., Oregon, Frankfurt).
   - **Branch**: `main`
   - **Language**: **Docker** (Render will automatically detect the root `Dockerfile`).
   - **Plan**: Select **Free** (0.1 CPU, 512 MB RAM).
5. Leave the default environment variables as-is. Render automatically sets `PORT=10000`, which our `CMD` evaluates.
6. Click **Create Web Service**.

### Step 3: Monitor Build and Deployment
1. Render will stream the build log:
   - Installing `tesseract-ocr` via `apt-get`
   - Installing Python dependencies from `backend/requirements.txt`
   - Starting `uvicorn app.main:app --host 0.0.0.0 --port 10000`
2. Once deployed, Render provides an assigned URL:
   `https://docusense-ai-xxxx.onrender.com`

---

## Local Docker Testing (When Docker is Installed)

To test the container locally on any system with Docker installed:

```bash
# 1. Build the Docker image
docker build -t docusense-ai .

# 2. Run the container locally mapping host port 7860 to container port 7860
docker run -p 7860:7860 --name docusense-container docusense-ai

# 3. Test with a custom port if desired (e.g. port 8000)
docker run -e PORT=8000 -p 8000:8000 docusense-ai
```

Open `http://localhost:7860` in your web browser.

---

## Post-Deployment Verification Checklist

After deploying to Hugging Face Spaces or Render, verify all critical application pathways:

### 1. Health & Tesseract Availability
Open `https://<YOUR-DEPLOYED-URL>/api/health` in your browser.
Expected response:
```json
{
  "status": "ok",
  "tesseract_available": true,
  "vision_model": "not_run"
}
```
- Verify that `"tesseract_available"` is `true`. If `false`, Tesseract binary was not found.

### 2. Web UI Loading
Open `https://<YOUR-DEPLOYED-URL>/` in your browser.
- Verify the DocuSense AI dashboard loads with the upload zone, currency selector, and field table.

### 3. Image Document Upload & OCR
Upload a standard PNG or JPEG receipt or invoice.
- Verify that upload succeeds with status 200.
- Verify that OCR extracts text and populates fields (`vendor`, `document_number`, `date`, `subtotal`, `tax`, `total`).
- Verify confidence indicators and status badges (`VALID`, `MISSING`, `AMBIGUOUS`, etc.) render correctly.

### 4. Digital PDF Document Upload
Upload a multi-page or digital PDF document.
- Verify that `source` reports `"pdf_text"` (or `"ocr"` if scanned).
- Verify arithmetic validation (subtotal + tax vs total) runs accurately.

### 5. Manual Field Correction
Click to edit an ambiguous or missing field and save.
- Verify that the correction updates the status to `VALID` and marks the field's source as `manual`.
- Verify arithmetic status recalculates dynamically if monetary values change.

### 6. Export Functionality
- Click **Export JSON** -> confirm valid JSON file downloads with all fields, confidence scores, and flags.
- Click **Export CSV** -> confirm CSV downloads with sanitized formula injection cells (preventing `=+-@` injection).

### 7. Ephemeral Disk Resilience
Trigger a container restart (or wait for Render's 15-minute sleep cycle and wake it up).
- Verify that the application boots cleanly even if previous uploads were wiped from disk.
- Verify that the app serves the homepage and accepts new document uploads without crashing.
