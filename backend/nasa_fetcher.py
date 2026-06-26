"""
LunarSpec — NASA PDS Data Fetcher
Downloads real lunar hyperspectral and image data from NASA archives.

Usage:
  python nasa_fetcher.py --list          # list available datasets
  python nasa_fetcher.py --download m3   # download Chandrayaan M3 sample
  python nasa_fetcher.py --download lroc # download LRO NAC sample
  python nasa_fetcher.py --download diviner # download thermal data
"""

import os, sys, argparse, hashlib
from pathlib import Path
import urllib.request
import urllib.error

DOWNLOAD_DIR = Path("./nasa_data")
DOWNLOAD_DIR.mkdir(exist_ok=True)

# ── REAL NASA PDS DATASETS ────────────────────────────────
# These are actual publicly available URLs from NASA PDS
DATASETS = {
    # Chandrayaan-1 Moon Mineralogy Mapper (M3) — hyperspectral
    # Full archive: https://pds-imaging.jpl.nasa.gov/volumes/m3.html
    "m3": {
        "name": "Chandrayaan M3 Hyperspectral (South Pole sample)",
        "description": "Moon Mineralogy Mapper — 85 spectral bands 0.43–3.0µm, 140m/px",
        "files": [
            {
                "url": "https://pds-imaging.jpl.nasa.gov/data/m3/CH1M3_0004/DATA/20090501_20090515/200905/L1B/M3G20090507T040833_V03_RDN.IMG",
                "filename": "m3_south_pole_rdn.img",
                "type": "hyperspectral",
                "note": "Radiance cube — run through spectral matcher directly"
            },
            {
                "url": "https://pds-imaging.jpl.nasa.gov/data/m3/CH1M3_0004/DATA/20090501_20090515/200905/L1B/M3G20090507T040833_V03_RDN.LBL",
                "filename": "m3_south_pole_rdn.lbl",
                "type": "label",
                "note": "PDS label — contains band wavelength info"
            }
        ]
    },

    # LRO NAC — high resolution grayscale images
    # Full archive: https://pds.lroc.asu.edu/
    "lroc": {
        "name": "LRO NAC Mosaic (South Pole)",
        "description": "Lunar Reconnaissance Orbiter Camera — 0.5m/px grayscale",
        "files": [
            {
                "url": "https://pds.lroc.asu.edu/data/LRO-L-LROC-2-EDR-V1.0/LROLRC_0001/DATA/ESE/2009284/NAC/M102285549LE.IMG",
                "filename": "lroc_nac_south_pole.img",
                "type": "rgb",
                "note": "Single-band NAC image"
            }
        ]
    },

    # LRO Diviner — thermal infrared
    # Full archive: https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/
    "diviner": {
        "name": "LRO Diviner Thermal RDR",
        "description": "Diviner Lunar Radiometer — surface temperature, 7 thermal bands",
        "files": [
            {
                "url": "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1001/data/2009/275/dlre_rdr_2009275_0000.tab",
                "filename": "diviner_thermal_2009.tab",
                "type": "thermal",
                "note": "Tab-delimited thermal data — temperature + coords"
            }
        ]
    },

    # Kaguya (SELENE) Spectral Profiler — small sample
    "kaguya": {
        "name": "Kaguya SELENE Spectral Profiler",
        "description": "JAXA SELENE SP — 296 bands 0.5–2.6µm",
        "files": [
            {
                "url": "https://darts.jaxa.jp/pub/selene/SP/SP_Level2C/SP_Level2C_v02.2/2008/01/SEL_SP2C_RBR_20080101_014534_001_V02.2.lbl",
                "filename": "kaguya_sp_sample.lbl",
                "type": "hyperspectral",
                "note": "Label file — fetch corresponding .img for data"
            }
        ]
    },

    # Test image — small synthetic GeoTIFF for pipeline testing
    "test": {
        "name": "Synthetic test GeoTIFF (generated locally)",
        "description": "Small 256×256 5-band synthetic image for pipeline testing",
        "files": []  # generated locally, see generate_test_image()
    }
}


def list_datasets():
    print("\n╔══ Available NASA datasets ══════════════════════════════╗")
    for key, ds in DATASETS.items():
        print(f"\n  [{key}]  {ds['name']}")
        print(f"         {ds['description']}")
        for f in ds.get('files', []):
            print(f"         → {f['filename']}  ({f['type']})  {f.get('note','')}")
    print("\n  Use: python nasa_fetcher.py --download <key>\n")


def download_dataset(key: str):
    if key == "test":
        generate_test_image()
        return

    if key not in DATASETS:
        print(f"Unknown dataset '{key}'. Use --list to see options.")
        sys.exit(1)

    ds = DATASETS[key]
    print(f"\nDownloading: {ds['name']}")

    for f in ds["files"]:
        dest = DOWNLOAD_DIR / f["filename"]
        if dest.exists():
            print(f"  ✓ Already downloaded: {dest}")
            continue

        print(f"  ↓ {f['url']}")
        print(f"    → {dest}")

        try:
            req = urllib.request.Request(f["url"], headers={"User-Agent": "LunarSpec/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                with open(dest, "wb") as out:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        out.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            print(f"\r    {pct:.1f}% ({downloaded/1_048_576:.1f} MB)", end="", flush=True)
            print(f"\n  ✓ Saved: {dest} ({dest.stat().st_size/1_048_576:.1f} MB)")

        except urllib.error.HTTPError as e:
            print(f"\n  ✗ HTTP {e.code}: {e.reason}")
            print(f"    The file may have moved. Browse the archive manually:")
            print(f"    https://pds-imaging.jpl.nasa.gov/")
        except Exception as e:
            print(f"\n  ✗ Error: {e}")


def generate_test_image():
    """Generate a small synthetic hyperspectral GeoTIFF for pipeline testing."""
    print("\nGenerating synthetic test image...")
    try:
        import numpy as np
        try:
            import rasterio
            from rasterio.transform import from_bounds
            from rasterio.crs import CRS

            H, W, B = 256, 256, 20
            # South pole bounding box
            transform = from_bounds(-5, -90, 5, -85, W, H)

            # Simulate spectral zones
            data = np.zeros((B, H, W), dtype=np.float32)
            wavelengths = np.linspace(0.4, 2.5, B)

            for b, wl in enumerate(wavelengths):
                # Pyroxene zone (top-left): absorption at 1.0µm and 2.0µm
                zone1 = np.zeros((H, W))
                zone1[:H//2, :W//2] = 0.3 - 0.15 * np.exp(-((wl-1.0)**2)/0.05) - 0.12 * np.exp(-((wl-2.0)**2)/0.05)

                # Olivine zone (top-right): absorption at 1.05µm
                zone2 = np.zeros((H, W))
                zone2[:H//2, W//2:] = 0.25 - 0.18 * np.exp(-((wl-1.05)**2)/0.06)

                # Plagioclase (bottom-left): flat bright
                zone3 = np.zeros((H, W))
                zone3[H//2:, :W//2] = 0.45 + np.random.uniform(-0.02, 0.02)

                # PSR / ice zone (bottom-right): absorption at 1.5 + 2.0µm
                zone4 = np.zeros((H, W))
                zone4[H//2:, W//2:] = 0.20 - 0.20 * np.exp(-((wl-1.5)**2)/0.04) - 0.18 * np.exp(-((wl-2.0)**2)/0.04)

                data[b] = zone1 + zone2 + zone3 + zone4
                data[b] += np.random.normal(0, 0.01, (H, W))  # noise
                data[b] = np.clip(data[b], 0, 1)

            out_path = DOWNLOAD_DIR / "synthetic_test_hyperspectral.tif"
            with rasterio.open(
                out_path, 'w',
                driver='GTiff',
                height=H, width=W,
                count=B,
                dtype='float32',
                crs=CRS.from_epsg(4326),
                transform=transform
            ) as dst:
                dst.write(data)
                # Write wavelength metadata per band
                for i, wl in enumerate(wavelengths):
                    dst.update_tags(i+1, wavelength=str(round(float(wl), 4)))

            print(f"  ✓ Saved: {out_path}")
            print(f"    {B} bands, {W}×{H}px, south pole bbox")
            print(f"    Mineral zones: pyroxene (NW), olivine (NE), plagioclase (SW), ice/PSR (SE)")
            print(f"\n  Upload this file to LunarSpec to test the full pipeline.")

        except ImportError:
            # rasterio not available — generate simple PNG
            import struct, zlib

            print("  rasterio not installed — generating PNG fallback")
            H, W = 256, 256
            img = np.zeros((H, W, 3), dtype=np.uint8)
            img[:H//2, :W//2] = [180, 120, 60]   # pyroxene
            img[:H//2, W//2:] = [60, 160, 140]   # olivine
            img[H//2:, :W//2] = [200, 200, 220]  # plagioclase
            img[H//2:, W//2:] = [40, 80, 180]    # ice

            from PIL import Image
            out_path = DOWNLOAD_DIR / "synthetic_test.png"
            Image.fromarray(img).save(out_path)
            print(f"  ✓ Saved: {out_path} (3-band RGB — limited spectral analysis)")

    except Exception as e:
        print(f"  ✗ Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LunarSpec NASA data fetcher")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--download", metavar="KEY", help="Download dataset by key")
    args = parser.parse_args()

    if args.list:
        list_datasets()
    elif args.download:
        download_dataset(args.download)
    else:
        parser.print_help()
