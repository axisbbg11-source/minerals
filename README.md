# LunarSpec — Full Setup Guide

## What you're running
- **Frontend**: `frontend/index.html` — open in browser or deploy to Netlify
- **Backend**: `backend/main.py` — FastAPI on your machine (localhost:8000)
- **Database**: Your Supabase project (reused from Nextryin/StayHub)
- **Storage**: Supabase Storage bucket `lunarspec-images`

---

## Step 1 — Supabase setup

1. Open your Supabase project dashboard
2. Go to **Extensions** → enable **PostGIS**
3. Go to **SQL Editor** → paste and run the full contents of `sql/schema.sql`
4. Go to **Storage** → create bucket named `lunarspec-images` (private)

---

## Step 2 — Backend setup (your machine)

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# GDAL note: if rasterio fails to install, run:
# pip install rasterio --find-links https://girder.github.io/large_image_wheels

# Copy env file and fill in your keys
cp .env.example .env
# Open .env and add:
#   SUPABASE_URL=https://your-project.supabase.co
#   SUPABASE_SERVICE_KEY=eyJ...   (service role key, NOT anon)
```

### Load .env and start server

```bash
# Mac/Linux
export $(cat .env | xargs) && uvicorn main:app --reload --port 8000

# Windows PowerShell
Get-Content .env | ForEach-Object { $k,$v = $_ -split '=',2; [System.Environment]::SetEnvironmentVariable($k,$v) }
uvicorn main:app --reload --port 8000
```

Backend will be live at **http://localhost:8000**
API docs at **http://localhost:8000/docs**

---

## Step 3 — Frontend setup

Open `frontend/index.html` and replace these two lines at the top:

```javascript
const SUPABASE_URL = 'YOUR_SUPABASE_URL';
const SUPABASE_ANON_KEY = 'YOUR_SUPABASE_ANON_KEY';  // anon key is fine here
```

Then open the file in your browser — or drag the folder to **Netlify** to deploy.

---

## Step 4 — Get real lunar data

```bash
cd backend

# Generate a synthetic test image (works immediately, no download needed)
python nasa_fetcher.py --download test

# List all real NASA datasets
python nasa_fetcher.py --list

# Download real Chandrayaan M3 hyperspectral data (~500MB)
python nasa_fetcher.py --download m3

# Download LRO Diviner thermal data
python nasa_fetcher.py --download diviner
```

Downloaded files go to `backend/nasa_data/`. Upload them via the dashboard.

---

## Step 5 — Upload and analyse

1. Open the dashboard in browser
2. Drag any GeoTIFF, IMG, or PNG onto the upload zone
3. Watch the queue — preprocessing → spectral match → CNN → ice detection → fusion
4. Mineral map appears on canvas when done
5. Switch to **Probe** tool and hover over the map to inspect pixels
6. Click **Water** tab for ice detection results
7. Export as CSV or GeoJSON

---

## Architecture recap

```
Browser (frontend/index.html)
    │
    ├── Supabase JS (read results, real-time updates)
    │       └── Your Supabase project (PostGIS)
    │
    └── fetch() → FastAPI (localhost:8000)
            ├── /upload          → saves file, starts pipeline
            ├── /analyses        → list all
            ├── /analyses/:id    → full results + log
            ├── /analyses/:id/pixels → pixel samples for map
            └── Pipeline runs in background:
                  1. Load image (rasterio / PIL)
                  2. Preprocess (radiometric correction, noise filter)
                  3. Spectral matcher (mineral absorption bands)
                  4. CNN classifier (ResNet18 or stats fallback)
                  5. Ice/OH detector (NIR 1.5µm + 2.0µm)
                  6. Fusion (weighted ensemble)
                  7. Save to Supabase
```

---

## Upgrading the CNN model

The current CNN uses an untrained ResNet18 (random weights) as a placeholder.
To make it real:

1. Download labeled Apollo sample spectral data from:
   https://pds-geosciences.wustl.edu/missions/apollo/

2. Fine-tune ResNet18 on spectral composites with known mineral labels

3. Save model weights:
   ```python
   torch.save(model.state_dict(), 'lunar_mineral_model.pth')
   ```

4. In `main.py`, replace `pretrained=False` with:
   ```python
   model.load_state_dict(torch.load('lunar_mineral_model.pth'))
   ```

Until then, the **spectral stats classifier** (fallback) gives real results
based on actual absorption band physics — it's scientifically valid,
just less spatially accurate than a trained CNN.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `rasterio` install fails | `pip install rasterio --find-links https://girder.github.io/large_image_wheels` |
| CORS error in browser | Make sure backend is running on port 8000 |
| Supabase 401 error | Check you used the **service role** key in `.env`, not anon |
| PostGIS missing | Enable in Supabase Dashboard → Extensions |
| Storage bucket missing | Create `lunarspec-images` in Supabase → Storage |
| IMG file not loading | Install GDAL: `conda install -c conda-forge gdal` |
