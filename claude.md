# GeoRecon AI — Single-Pass Drone Video → Georeferenced 3D Model

SIH 2026 · PS 26158 (NTRO) · Team Not So Clueless.
This is a time-boxed prototype (~7 hours). Optimise for a working, demonstrable
end-to-end pipeline with honest metrics — not for breadth or abstraction.

## What we are building

Input: drone video (1080p/4K) + GPS telemetry (DJI `.SRT` or CSV) + optional
camera intrinsics / RTK-corrected positions.

Output: a georeferenced, metrically scaled 3D point cloud and mesh, exported as
PLY, OBJ, GLB, LAS and GeoTIFF (DSM + colour raster), plus validation metrics
and a web viewer with distance / area / height measurement.

## Stack — do not add dependencies without asking

- Backend: Python 3.11, FastAPI, uvicorn, pydantic-settings
- SfM / MVS: **<FILL IN: `pycolmap-cuda12` (Linux/WSL2/Colab) OR `colmap` 4.x CLI (Windows CUDA build)>**
- CV / 3D: OpenCV, NumPy, SciPy, Open3D, trimesh
- Geo: pyproj, laspy, rasterio
- AI: ultralytics (YOLO segmentation) for dynamic-object masks
- Frontend: React + Vite + TypeScript, @react-three/fiber + drei, three.js
  PLYLoader/GLTFLoader, react-leaflet, Tailwind
- Job state: filesystem + `status.json`. No database, no Redis/Celery, no auth.

## Repo layout

```
backend/
  georecon/
    config.py            # presets + settings
    pipeline.py          # stage runner: timing, caching, status updates
    cli.py               # python -m georecon.cli run ...
    ingest/telemetry.py  # SRT/CSV parsers + time interpolation
    ingest/video.py      # decoding helpers
    stages/keyframes.py  stages/masking.py  stages/sfm.py  stages/georef.py
    stages/dense.py      stages/mesh.py     stages/export.py stages/validate.py
  api/main.py            # FastAPI app
  api/jobs.py            # background job runner (thread/process pool)
  tests/
frontend/
data/samples/            # full sample clip + telemetry
data/samples/tiny/       # 20–30 s fixture for fast iteration
jobs/<job_id>/           # per-job workspace (gitignored)
```

## Job workspace contract

```
jobs/<id>/
  input/        video.*, telemetry.*
  frames/       keyframes (jpg), frames.csv (name,t,lat,lon,alt,sharpness)
  masks/        <frame_name>.png  (COLMAP convention: 0 = ignore)
  sfm/          database.db, sparse/
  georef/       transform.json (s, R, t), origin.json (lat,lon,alt,utm_epsg)
  dense/        fused.ply (+ fused.ply.vis if produced)
  outputs/      pointcloud.ply, pointcloud.las, mesh.obj, model.glb,
                dsm.tif, color.tif, trajectory.geojson,
                web/pointcloud.ply (downsampled), web/model.glb
  report/       metrics.json
  status.json   {stage, progress, message, state: queued|running|done|failed, warnings[]}
  log.txt
```

## Stage rules

- Each stage is a function `run(ctx: StageContext) -> dict` that writes its outputs and
  returns metrics. The runner merges metrics + wall time into `report/metrics.json`.
- Stages are idempotent: if outputs exist and `force` is false, skip. This is how
  we iterate quickly — never recompute SfM just to tweak meshing.
- Update `status.json` atomically (write temp file, then rename).
- Log to `log.txt` and stdout. Fail loudly with a clear message in `status.json`.
- Graceful degradation, always with a warning in `status.json`:
  no telemetry → unscaled model; no CUDA → skip dense, mesh from sparse points.

## Coordinate conventions (important)

- Processing and web assets use **local ENU in metres**, origin = GPS of the first
  registered keyframe (stored in `georef/origin.json`).
- LAS and GeoTIFF exports use **UTM**, EPSG chosen from longitude (Pune → 32643).
- Never ship UTM/ECEF coordinates to the browser (float32 jitter). The frontend
  converts clicked ENU points back to lat/lon using `origin.json`.
- COLMAP camera centre is `C = -Rᵀ t`. The pycolmap API changed between 3.x and
  4.x — check names against the installed version
  (`python -c "import pycolmap; help(pycolmap.Image)"`) before writing code.
- Camera model: **SIMPLE_RADIAL** (`f, cx, cy, k`). More stable than OPENCV for a
  low-parallax single straight pass; upgrade only if parallax is clearly rich.

## Georeferencing — do not "simplify" this

1. Pair SfM camera centres with GPS (converted to ENU) for registered frames.
2. PCA of the GPS track. If well spread (σ2/σ1 ≥ 0.1): robust Umeyama sim(3)
   with RANSAC (inlier threshold ~5 m), then refine on inliers.
3. If near-collinear (σ2/σ1 < 0.1 — the normal single straight pass): roll about
   the flight line is unobservable from positions alone. Instead fit the ground
   plane to sparse points (RANSAC), rotate so its normal is +Z with cameras above
   it, solve a 2D similarity in XY (Umeyama 2D, det > 0) for scale/yaw/shift,
   and set the Z offset by median residual.
4. Apply the transform with NumPy to cameras and point clouds (API-independent).
5. Report fit RMSE (horizontal and vertical separately) and 5-fold hold-out RMSE.

## Presets

| preset   | max keyframes | frame long side | MVS max_image_size |
|----------|---------------|-----------------|--------------------|
| fast     | 150           | 1280            | 800                |
| balanced | 300           | 1600            | 1200               |
| accurate | 600           | 2000            | 1600               |

## API contract

- `POST /api/jobs` multipart: `video`, `telemetry` (optional), `preset`, `mask_dynamic` → `{job_id}`
- `GET /api/jobs` → list with state + created time
- `GET /api/jobs/{id}` → status.json + metrics.json merged
- `GET /api/jobs/{id}/files/{path}` → serve files from `outputs/` and `report/` only (block path traversal)
- `GET /api/jobs/{id}/log?tail=200`

## Commands

- `python -m georecon.cli run --video V --telemetry T --job NAME --preset fast [--force-from STAGE]`
- `uvicorn api.main:app --reload --port 8000` (from `backend/`)
- `pytest -q` (from `backend/`)
- `npm run dev` (from `frontend/`, proxies `/api` to :8000)

## Working rules for Claude

- Prototype under a hard deadline: minimal dependencies, no speculative abstractions.
- After writing code, RUN it on `data/samples/tiny` and show evidence: files written,
  counts, and metrics. Use the full sample only at milestones.
- Do not refactor working modules unless asked. Keep changes scoped to the task.
- If an install or build problem takes more than ~10 minutes, stop and report
  options instead of fighting it.
- Suggest a git commit message after each working step.