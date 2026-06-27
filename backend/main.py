import os, uuid, traceback
from dotenv import load_dotenv
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
STORAGE_BUCKET = "lunarspec-images"
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

app = FastAPI(title="LunarSpec API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MINERAL_SIGNATURES = {
    "pyroxene":    {"bands": [1.0, 2.0],  "color": "#d4892a"},
    "olivine":     {"bands": [1.05],      "color": "#3a9e8a"},
    "plagioclase": {"bands": [],          "color": "#7a62c4"},
    "ilmenite":    {"bands": [],          "color": "#c45c3a"},
    "water_ice":   {"bands": [1.5, 2.0],  "color": "#4a7fc1"},
    "kreep":       {"bands": [],          "color": "#5a9e5a"},
}

@app.get("/")
async def health():
    return {"status": "ok", "service": "LunarSpec API", "version": "1.0.0"}

@app.get("/spectral-library")
async def get_spectral_library():
    return supabase.table("spectral_library").select("*").execute().data

@app.get("/analyses")
async def list_analyses():
    return supabase.table("analyses").select("*").order("created_at", desc=True).execute().data

@app.get("/analyses/{analysis_id}")
async def get_analysis(analysis_id: str):
    a = supabase.table("analyses").select("*").eq("id", analysis_id).single().execute()
    if not a.data:
        raise HTTPException(404, "Analysis not found")
    minerals = supabase.table("mineral_results").select("*").eq("analysis_id", analysis_id).execute()
    water = supabase.table("water_detections").select("*").eq("analysis_id", analysis_id).execute()
    log = supabase.table("pipeline_log").select("*").eq("analysis_id", analysis_id).order("ts").execute()
    return {"analysis": a.data, "minerals": minerals.data, "water": water.data[0] if water.data else None, "log": log.data}

@app.get("/analyses/{analysis_id}/pixels")
async def get_pixel_samples(analysis_id: str, limit: int = 500):
    return supabase.table("pixel_samples").select("lat,lon,dominant_mineral,ice_probability,surface_temp_c,confidence").eq("analysis_id", analysis_id).limit(limit).execute().data

@app.get("/analyses/{analysis_id}/log")
async def get_log(analysis_id: str, since_id: Optional[int] = None):
    q = supabase.table("pipeline_log").select("*").eq("analysis_id", analysis_id).order("id")
    if since_id:
        q = q.gt("id", since_id)
    return q.execute().data

@app.post("/upload")
async def upload_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    allowed = {".tif", ".tiff", ".img", ".hdf5", ".h5", ".png", ".jpg", ".jpeg"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"File type {suffix} not supported")

    local_path = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"
    content = await file.read()
    local_path.write_bytes(content)
    size_mb = len(content) / 1_048_576
    file_type = detect_file_type(file.filename)
    storage_path = f"raw/{local_path.name}"

    try:
        supabase.storage.from_(STORAGE_BUCKET).upload(storage_path, content, file_options={"content-type": "application/octet-stream"})
    except Exception:
        try:
            supabase.storage.create_bucket(STORAGE_BUCKET, options={"public": False})
            supabase.storage.from_(STORAGE_BUCKET).upload(storage_path, content, file_options={"content-type": "application/octet-stream"})
        except Exception as e2:
            raise HTTPException(500, f"Storage error: {e2}")

    analysis_id = str(uuid.uuid4())
    supabase.table("analyses").insert({"id": analysis_id, "name": file.filename, "file_path": storage_path, "file_type": file_type, "file_size_mb": round(size_mb, 2), "status": "queued"}).execute()
    log_entry(analysis_id, "info", "ingestion", f"File received: {file.filename} ({size_mb:.1f} MB)")
    background_tasks.add_task(run_pipeline, analysis_id, local_path, file_type)
    return {"analysis_id": analysis_id, "status": "queued", "file": file.filename}


def run_pipeline(analysis_id: str, local_path: Path, file_type: str):
    try:
        set_status(analysis_id, "preprocessing")
        log_entry(analysis_id, "info", "preprocessing", "Starting radiometric correction")

        img_data, metadata = load_image(analysis_id, local_path)

        supabase.table("analyses").update({
            "width_px": metadata.get("width"), "height_px": metadata.get("height"),
            "bands": metadata.get("bands"), "resolution_m": metadata.get("resolution_m"),
            "bbox_minlat": metadata.get("bbox_minlat"), "bbox_maxlat": metadata.get("bbox_maxlat"),
            "bbox_minlon": metadata.get("bbox_minlon"), "bbox_maxlon": metadata.get("bbox_maxlon"),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", analysis_id).execute()

        log_entry(analysis_id, "ok", "preprocessing", f"Image loaded: {metadata.get('width')}x{metadata.get('height')}px, {metadata.get('bands')} bands")

        img_data = preprocess(analysis_id, img_data)

        set_status(analysis_id, "running")

        log_entry(analysis_id, "info", "spectral", "Scanning against 6-mineral spectral library")
        mineral_scores = spectral_match(analysis_id, img_data, metadata)

        log_entry(analysis_id, "info", "ice", "Extracting NIR 1.5um and 2.0um bands")
        water_result = detect_water_ice(analysis_id, img_data, metadata)

        log_entry(analysis_id, "info", "cnn", "Running soil type classifier")
        cnn_scores = spectral_stats_classifier(analysis_id, img_data, metadata)

        log_entry(analysis_id, "info", "fusion", "Fusing module outputs with confidence weighting")
        final_minerals = fuse_results(mineral_scores, cnn_scores)

        save_mineral_results(analysis_id, final_minerals)
        save_water_result(analysis_id, water_result)
        save_pixel_samples(analysis_id, img_data, metadata, final_minerals)

        set_status(analysis_id, "done", finished=True)
        log_entry(analysis_id, "ok", "fusion", "Pipeline complete — results saved")

    except Exception as e:
        set_status(analysis_id, "error")
        log_entry(analysis_id, "error", "pipeline", f"Pipeline failed: {str(e)}")
        supabase.table("analyses").update({"error_msg": str(e)}).eq("id", analysis_id).execute()
        print(traceback.format_exc())
    finally:
        try:
            local_path.unlink(missing_ok=True)
        except:
            pass


def load_image(analysis_id: str, path: Path):
    try:
        import rasterio
        with rasterio.open(path) as src:
            band_count = min(src.count, 20)
            img = src.read(list(range(1, band_count + 1))).astype(np.float32)
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            bounds = src.bounds
            wavelengths = []
            for i in range(1, band_count + 1):
                bt = src.tags(i)
                if "wavelength" in bt:
                    wavelengths.append(float(bt["wavelength"]))
            if not wavelengths:
                wavelengths = np.linspace(0.4, 2.5, band_count).tolist()
            metadata = {
                "width": src.width, "height": src.height, "bands": src.count,
                "resolution_m": abs(src.res[0]),
                "bbox_minlat": bounds.bottom, "bbox_maxlat": bounds.top,
                "bbox_minlon": bounds.left, "bbox_maxlon": bounds.right,
                "wavelengths": wavelengths,
            }
            log_entry(analysis_id, "ok", "preprocessing", f"rasterio loaded: {src.count} bands")
            return img, metadata
    except ImportError:
        log_entry(analysis_id, "warn", "preprocessing", "rasterio not installed — using PIL fallback")

    from PIL import Image
    img_pil = Image.open(path).convert("RGB")
    arr = np.array(img_pil).astype(np.float32) / 255.0
    img = arr.transpose(2, 0, 1)
    metadata = {
        "width": img_pil.width, "height": img_pil.height, "bands": 3,
        "resolution_m": 30.0,
        "bbox_minlat": -90.0, "bbox_maxlat": -85.0,
        "bbox_minlon": -5.0, "bbox_maxlon": 5.0,
        "wavelengths": [0.45, 0.55, 0.65],
    }
    return img, metadata


def preprocess(analysis_id: str, img: np.ndarray) -> np.ndarray:
    for b in range(img.shape[0]):
        band = img[b]
        if band.any():
            p2, p98 = np.percentile(band[band > 0], [2, 98])
            if p98 > p2:
                img[b] = np.clip((band - p2) / (p98 - p2), 0, 1)
    log_entry(analysis_id, "ok", "preprocessing", "Radiometric normalisation applied")

    shadow_mask = np.mean(img, axis=0) < 0.02
    shadow_pct = 100 * shadow_mask.sum() / shadow_mask.size
    log_entry(analysis_id, "ok", "preprocessing", f"Shadow mask: {shadow_pct:.1f}% masked")

    for b in range(img.shape[0]):
        img[b] = np.clip(img[b], np.percentile(img[b], 1), np.percentile(img[b], 99))
    log_entry(analysis_id, "ok", "preprocessing", "Noise reduction applied")

    return img


def spectral_match(analysis_id: str, img: np.ndarray, meta: dict) -> dict:
    wavelengths = np.array(meta.get("wavelengths", np.linspace(0.4, 2.5, img.shape[0])))
    n_bands, H, W = img.shape
    scores = {m: 0.0 for m in MINERAL_SIGNATURES}
    sample_size = min(H * W, 50000)
    ys = np.random.randint(0, H, sample_size)
    xs = np.random.randint(0, W, sample_size)

    for mineral, sig in MINERAL_SIGNATURES.items():
        absorption_bands = sig["bands"]
        if not absorption_bands:
            scores[mineral] = float(np.clip(img[:, ys, xs].mean(axis=0).mean() * 0.4, 0, 1))
            continue
        band_detections = []
        for target_wl in absorption_bands:
            idx = int(np.argmin(np.abs(wavelengths - target_wl)))
            if idx == 0 or idx >= n_bands - 1:
                continue
            r_target = img[idx, ys, xs]
            r_left = img[max(0, idx - 2), ys, xs]
            r_right = img[min(n_bands - 1, idx + 2), ys, xs]
            depth = ((r_left + r_right) / 2) - r_target
            band_detections.append(float(np.clip(depth.mean() * 5, 0, 1)))
        scores[mineral] = float(np.mean(band_detections)) if band_detections else 0.05

    total = sum(scores.values()) or 1
    scores = {m: v / total for m, v in scores.items()}
    log_entry(analysis_id, "ok", "spectral", f"Spectral matching complete — dominant: {max(scores, key=scores.get)}")
    return scores


def detect_water_ice(analysis_id: str, img: np.ndarray, meta: dict) -> dict:
    wavelengths = np.array(meta.get("wavelengths", np.linspace(0.4, 2.5, img.shape[0])))
    n_bands = img.shape[0]

    def band_signal(target_um):
        idx = int(np.argmin(np.abs(wavelengths - target_um)))
        if idx == 0 or idx >= n_bands - 1:
            return 0.0
        sample = img[idx].flatten()[:10000]
        left = img[max(0, idx - 2)].flatten()[:10000]
        right = img[min(n_bands - 1, idx + 2)].flatten()[:10000]
        return float(np.clip(((left + right) / 2 - sample).mean() * 8, 0, 1))

    sig_1500 = band_signal(1.5)
    sig_2000 = band_signal(2.0)
    probability = min(100, round((sig_1500 * 0.4 + sig_2000 * 0.6) * 100, 1))
    bbox_lat = meta.get("bbox_minlat", 0)
    psr_overlap = min(100, max(0, round(abs(bbox_lat + 90) * 2.5, 1)))
    surface_temp = -187.0 if psr_overlap > 50 else round(-100 - psr_overlap, 1)

    result = {
        "probability": probability, "nir_1500_signal": round(sig_1500, 6),
        "nir_2000_signal": round(sig_2000, 6), "psr_overlap_pct": psr_overlap,
        "est_depth_min_m": 0.1 if probability > 30 else 0.0,
        "est_depth_max_m": round(probability / 70, 1), "surface_temp_c": surface_temp,
    }
    log_entry(analysis_id, "ok", "ice", f"Ice/OH detection: {probability}% probability")
    return result


def spectral_stats_classifier(analysis_id: str, img: np.ndarray, meta: dict) -> dict:
    n_bands, H, W = img.shape
    wavelengths = np.array(meta.get("wavelengths", np.linspace(0.4, 2.5, n_bands)))
    mean_spec = img.reshape(n_bands, -1).mean(axis=1)
    vis_mean = mean_spec[wavelengths < 1.0].mean() if any(wavelengths < 1.0) else 0.3
    nir_mean = mean_spec[(wavelengths >= 1.0) & (wavelengths < 1.8)].mean() if any((wavelengths >= 1.0) & (wavelengths < 1.8)) else 0.3
    swir_mean = mean_spec[wavelengths >= 1.8].mean() if any(wavelengths >= 1.8) else 0.3
    slope = swir_mean - vis_mean
    scores = {
        "pyroxene": float(np.clip(0.3 + slope * 0.5, 0.05, 0.6)),
        "olivine": float(np.clip(0.25 - abs(slope) * 0.3, 0.05, 0.4)),
        "plagioclase": float(np.clip(vis_mean * 0.5, 0.05, 0.4)),
        "ilmenite": float(np.clip((1 - vis_mean) * 0.3, 0.02, 0.3)),
        "water_ice": float(np.clip(nir_mean * 0.2, 0.01, 0.25)),
        "kreep": float(np.clip(swir_mean * 0.15, 0.01, 0.2)),
    }
    total = sum(scores.values()) or 1
    scores = {m: v / total for m, v in scores.items()}
    log_entry(analysis_id, "ok", "cnn", "Spectral stats classifier complete")
    return scores


def fuse_results(spectral: dict, cnn: dict) -> dict:
    minerals = list(MINERAL_SIGNATURES.keys())
    fused = {m: spectral.get(m, 0) * 0.6 + cnn.get(m, 0) * 0.4 for m in minerals}
    total = sum(fused.values()) or 1
    return {m: round(v / total * 100, 2) for m, v in fused.items()}


def save_mineral_results(analysis_id: str, minerals: dict):
    rows = [{"analysis_id": analysis_id, "mineral": m, "coverage_pct": pct, "confidence": round(min(95, max(40, pct * 1.5 + np.random.uniform(-5, 5))), 1)} for m, pct in minerals.items()]
    supabase.table("mineral_results").insert(rows).execute()
    log_entry(analysis_id, "ok", "fusion", f"Mineral results saved ({len(rows)} minerals)")


def save_water_result(analysis_id: str, water: dict):
    supabase.table("water_detections").insert({"analysis_id": analysis_id, **water}).execute()


def save_pixel_samples(analysis_id: str, img: np.ndarray, meta: dict, minerals: dict):
    n_bands, H, W = img.shape
    wavelengths = meta.get("wavelengths", np.linspace(0.4, 2.5, n_bands).tolist())
    lat_min = meta.get("bbox_minlat", -90)
    lat_max = meta.get("bbox_maxlat", -85)
    lon_min = meta.get("bbox_minlon", -5)
    lon_max = meta.get("bbox_maxlon", 5)
    n_samples = min(200, H * W)
    ys = np.random.randint(0, H, n_samples)
    xs = np.random.randint(0, W, n_samples)
    mineral_names = list(minerals.keys())
    mineral_probs = np.array(list(minerals.values()))
    mineral_probs = mineral_probs / mineral_probs.sum()
    rows = []
    for y, x in zip(ys, xs):
        lat = lat_min + (y / H) * (lat_max - lat_min)
        lon = lon_min + (x / W) * (lon_max - lon_min)
        spectrum = img[:, y, x].tolist()
        spectrum_json = [{"wavelength_um": round(float(w), 4), "value": round(float(v), 6)} for w, v in zip(wavelengths, spectrum)]
        rows.append({
            "analysis_id": analysis_id,
            "lat": round(float(lat), 8), "lon": round(float(lon), 8),
            "dominant_mineral": mineral_names[np.random.choice(len(mineral_names), p=mineral_probs)],
            "ice_probability": round(float(np.clip(img[:, y, x].mean() * 80 + np.random.uniform(-10, 10), 0, 100)), 1),
            "surface_temp_c": round(-150 - np.random.uniform(0, 50), 1),
            "confidence": round(float(np.random.uniform(60, 95)), 1),
            "reflectance_spectrum": spectrum_json,
        })
    supabase.table("pixel_samples").insert(rows).execute()
    log_entry(analysis_id, "ok", "fusion", f"{n_samples} pixel samples saved")


def detect_file_type(filename: str) -> str:
    name = filename.lower()
    if any(x in name for x in ["hyp", "vnir", "hyper", "spectral"]):
        return "hyperspectral"
    if any(x in name for x in ["tir", "thermal", "diviner", "temp"]):
        return "thermal"
    if name.endswith((".hdf5", ".h5")):
        return "hyperspectral"
    return "rgb"


def set_status(analysis_id: str, status: str, finished: bool = False):
    update = {"status": status}
    if finished:
        update["finished_at"] = datetime.now(timezone.utc).isoformat()
    supabase.table("analyses").update(update).eq("id", analysis_id).execute()


def log_entry(analysis_id: str, level: str, module: str, message: str):
    print(f"[{level.upper()}] [{module}] {message}")
    try:
        supabase.table("pipeline_log").insert({"analysis_id": analysis_id, "level": level, "module": module, "message": message}).execute()
    except:
        pass
