# OSM → SUMO Bridge

## Architecture
Leaflet UI (GitHub Pages or local) → FastAPI (localhost:8000) → Overpass/OSMnx → netconvert →
DistanceMatrix.ai snapshots (SQLite) → BPR-inverted v/c → hourly profile → Poisson O/D trips +
randomTrips.py background → duarouter → tlsCycleAdaptation.py (Webster) → sumo → tripinfo stats.

## Run locally (Windows/Linux/macOS)
1. Install Python 3.10+ and (optional) SUMO from https://sumo.dlr.de (or rely on `pip install eclipse-sumo`).
   If using the installer, set env var SUMO_HOME and add SUMO's bin to PATH.
2. `cd backend && python -m venv .venv && .venv\Scripts\activate` (Linux/macOS: `source .venv/bin/activate`)
3. `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and add your DistanceMatrix.ai key.
5. `uvicorn main:app --reload --port 8000`
6. Serve the frontend: `cd .. && python -m http.server 5500` and open http://localhost:5500
7. Outputs land in `backend/jobs/<id>/` — open `run.sumocfg` in `sumo-gui`.

## Build historical baseline (free tier has no history)
Call `POST /api/snapshot` (body: `{"o":{"lat":..,"lon":..},"d":{"lat":..,"lon":..}}`) every 15–30 min via
Task Scheduler/cron for 7–30 days. Without snapshots the tool uses a default South Asian time-of-day profile.

## Push to GitHub Pages
```
git init && git add . && git commit -m "OSM-SUMO bridge"
git branch -M main
git remote add origin https://github.com/<you>/osm-sumo.git
git push -u origin main
```
Repo → Settings → Pages → Deploy from branch `main` / root. Add `backend/.env`, `jobs/`, `snapshots.db` to `.gitignore`.
Note: an https GitHub Pages site calling http://127.0.0.1 works in Chrome/Edge/Firefox (localhost is exempt from mixed-content blocking).
