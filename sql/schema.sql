-- ============================================================
-- LunarSpec — Supabase Schema
-- Run this in your Supabase SQL editor
-- Requires: PostGIS extension (enable in Extensions tab first)
-- ============================================================

-- Enable PostGIS for geospatial support
create extension if not exists postgis;

-- ── ANALYSES ──────────────────────────────────────────────
-- One row per image uploaded and analysed
create table if not exists analyses (
  id            uuid primary key default gen_random_uuid(),
  created_at    timestamptz default now(),
  name          text not null,                  -- original filename
  file_path     text not null,                  -- Supabase Storage path
  file_type     text not null,                  -- 'hyperspectral' | 'thermal' | 'rgb'
  file_size_mb  numeric(10,2),
  status        text not null default 'queued', -- queued | preprocessing | running | done | error
  error_msg     text,
  -- image metadata
  width_px      int,
  height_px     int,
  bands         int,
  resolution_m  numeric(10,2),                  -- metres per pixel
  -- bounding box (lat/lon)
  bbox_minlat   numeric(12,8),
  bbox_maxlat   numeric(12,8),
  bbox_minlon   numeric(12,8),
  bbox_maxlon   numeric(12,8),
  -- pipeline timing
  started_at    timestamptz,
  finished_at   timestamptz
);

-- ── MINERAL RESULTS ───────────────────────────────────────
-- Per-analysis aggregate mineral composition
create table if not exists mineral_results (
  id            uuid primary key default gen_random_uuid(),
  analysis_id   uuid references analyses(id) on delete cascade,
  mineral       text not null,   -- 'pyroxene' | 'olivine' | 'plagioclase' | 'ilmenite' | 'water_ice' | 'kreep'
  coverage_pct  numeric(5,2),    -- % of image covered
  confidence    numeric(5,2),    -- model confidence 0–100
  avg_reflectance numeric(8,6),
  created_at    timestamptz default now()
);

-- ── WATER ICE DETECTION ───────────────────────────────────
create table if not exists water_detections (
  id              uuid primary key default gen_random_uuid(),
  analysis_id     uuid references analyses(id) on delete cascade,
  probability     numeric(5,2),   -- 0–100
  nir_1500_signal numeric(8,6),   -- 1.5 µm absorption band value
  nir_2000_signal numeric(8,6),   -- 2.0 µm absorption band value
  psr_overlap_pct numeric(5,2),   -- % overlap with permanently shadowed regions
  est_depth_min_m numeric(6,2),
  est_depth_max_m numeric(6,2),
  surface_temp_c  numeric(7,2),
  created_at      timestamptz default now()
);

-- ── PIXEL SAMPLES ─────────────────────────────────────────
-- Sparse samples stored for probe tool and spectra chart
create table if not exists pixel_samples (
  id            uuid primary key default gen_random_uuid(),
  analysis_id   uuid references analyses(id) on delete cascade,
  lat           numeric(12,8) not null,
  lon           numeric(12,8) not null,
  location      geometry(Point, 4326),  -- PostGIS point
  dominant_mineral text,
  ice_probability  numeric(5,2),
  surface_temp_c   numeric(7,2),
  confidence       numeric(5,2),
  reflectance_spectrum jsonb,           -- array of {wavelength_um, value}
  created_at    timestamptz default now()
);

-- Spatial index for fast bbox queries
create index if not exists pixel_samples_location_idx
  on pixel_samples using gist(location);

-- ── MODULE LOG ────────────────────────────────────────────
create table if not exists pipeline_log (
  id          bigserial primary key,
  analysis_id uuid references analyses(id) on delete cascade,
  ts          timestamptz default now(),
  level       text not null,  -- 'info' | 'ok' | 'warn' | 'error'
  module      text,           -- 'preprocessing' | 'spectral' | 'cnn' | 'ice' | 'fusion'
  message     text not null
);

-- ── SPECTRAL LIBRARY ──────────────────────────────────────
-- Reference mineral absorption signatures
create table if not exists spectral_library (
  id          serial primary key,
  mineral     text not null unique,
  display_name text not null,
  color_hex   text not null,
  -- characteristic absorption bands (µm)
  band1_um    numeric(6,4),
  band2_um    numeric(6,4),
  band3_um    numeric(6,4),
  -- reference spectrum (sampled at 0.1µm intervals 0.4–2.5µm)
  spectrum    jsonb,
  notes       text
);

-- Seed spectral library with real mineral signatures
insert into spectral_library (mineral, display_name, color_hex, band1_um, band2_um, band3_um, notes) values
('pyroxene',    'Pyroxene',          '#d4892a', 1.0,  2.0,  null, 'Strong 1µm and 2µm absorption bands. Most common lunar mineral.'),
('olivine',     'Olivine',           '#3a9e8a', 1.05, null, null, 'Broad absorption near 1µm, no 2µm band. Indicates mantle material.'),
('plagioclase', 'Plagioclase feldspar','#7a62c4', null, null, null, 'Featureless flat spectrum in VNIR. Dominant in lunar highlands.'),
('ilmenite',    'Ilmenite (FeTiO₃)', '#c45c3a', null, null, null, 'Dark, low-reflectance mineral. Indicator of mare basalt.'),
('water_ice',   'Water / ice',       '#4a7fc1', 1.5,  2.0,  3.0,  'OH/H₂O absorption at 1.5µm and 2.0µm. Found in PSRs.'),
('kreep',       'KREEP terrain',     '#5a9e5a', null, null, null, 'High K, REE, P signature. Geochemically evolved terrain.')
on conflict (mineral) do nothing;

-- ── ROW LEVEL SECURITY ────────────────────────────────────
-- Simple: service role key has full access, anon can read
alter table analyses         enable row level security;
alter table mineral_results  enable row level security;
alter table water_detections enable row level security;
alter table pixel_samples    enable row level security;
alter table pipeline_log     enable row level security;
alter table spectral_library enable row level security;

create policy "Public read analyses"        on analyses         for select using (true);
create policy "Public read mineral_results" on mineral_results  for select using (true);
create policy "Public read water"           on water_detections for select using (true);
create policy "Public read pixel_samples"   on pixel_samples    for select using (true);
create policy "Public read log"             on pipeline_log     for select using (true);
create policy "Public read spectral_lib"    on spectral_library for select using (true);

-- Service role (backend) can do everything — granted automatically via service key
