"""OSM -> SUMO pipeline. Run: uvicorn main:app --reload --port 8000"""
import os, sys, json, math, uuid, random, sqlite3, subprocess, shutil, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd, requests, osmnx as ox
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()
KEY = os.getenv("DISTANCEMATRIX_KEY", "")
WORK, DB = Path("jobs"), "snapshots.db"
WORK.mkdir(exist_ok=True)
SUMO_HOME = os.environ.get("SUMO_HOME")
if not SUMO_HOME:
    import sumo; SUMO_HOME = sumo.SUMO_HOME          # pip eclipse-sumo
TOOLS = Path(SUMO_HOME) / "tools"
sys.path.append(str(TOOLS))
import sumolib

app = FastAPI(title="OSM-SUMO Bridge")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5500", "http://127.0.0.1:5500",
                   "https://*.github.io", "null"], allow_origin_regex=r"https://.*\.github\.io|http://(localhost|127\.0\.0\.1):\d+",
                   allow_methods=["*"], allow_headers=["*"])
JOBS: dict = {}

class Pt(BaseModel): lat: float; lon: float
class Req(BaseModel):
    bbox: list[float] = Field(..., description="[south, west, north, east]")
    origin: Pt; dest: Pt
    lanes: int = 2                       # lanes on the O/D corridor
    days: int = 7                        # aggregation window of stored snapshots
    optimize_signals: bool = True
    background_share: float = 0.35       # share of demand on unmeasured minor roads
    simulate: bool = True

# ---------- South Asian mixed-flow assumptions (tune with local counts!) ----------
HOURLY = np.array([.18,.12,.10,.10,.15,.30,.55,.80,1.0,.90,.80,.80,.85,.85,.80,.82,.90,1.0,.95,.80,.60,.45,.32,.24])
MIX = {"car":.30,"motorcycle":.35,"cng":.15,"bus":.08,"truck":.05,"rickshaw":.07}   # share of vehicles
VTYPES = """<additional>
<vType id="car" vClass="passenger" length="4.2" width="1.8" maxSpeed="16" accel="2.5" sigma="0.6" latAlignment="center" minGapLat="0.3"/>
<vType id="motorcycle" vClass="motorcycle" length="2.0" width="0.8" maxSpeed="17" accel="3.5" sigma="0.7" latAlignment="arbitrary" minGapLat="0.15" lcPushy="0.8"/>
<vType id="cng" vClass="passenger" length="2.6" width="1.4" maxSpeed="12" accel="2" sigma="0.7" latAlignment="arbitrary" minGapLat="0.2"/>
<vType id="bus" vClass="bus" length="11" width="2.5" maxSpeed="13" accel="1.2" sigma="0.5" latAlignment="right"/>
<vType id="truck" vClass="truck" length="8" width="2.5" maxSpeed="12" accel="1.0" sigma="0.5" latAlignment="right"/>
<vType id="rickshaw" vClass="bicycle" length="2.5" width="1.0" maxSpeed="4" accel="0.8" sigma="0.8" latAlignment="right"/>
</additional>"""

def run(cmd, cwd):
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if p.returncode: raise RuntimeError(f"{cmd[0]} failed:\n{p.stderr[-1500:]}")
    return p.stdout

# ---------- 1. OSM ----------
def fetch_osm(bbox, out: Path):
    s, w, n, e = bbox
    q = f'[out:xml][timeout:120];(way["highway"]({s},{w},{n},{e});>;);out meta;'
    r = requests.post("https://overpass-api.de/api/interpreter", data={"data": q}, timeout=180)
    r.raise_for_status(); out.write_bytes(r.content)

def corridor_info(bbox, o, d):
    """OSMnx shortest path -> length (m) and free-flow estimate; also capacity hints."""
    s, w, n, e = bbox
    G = ox.graph_from_bbox((w, s, e, n), network_type="drive")   # osmnx>=2: (left,bottom,right,top)
    G = ox.add_edge_speeds(G); G = ox.add_edge_travel_times(G)
    a, b = ox.nearest_nodes(G, o.lon, o.lat), ox.nearest_nodes(G, d.lon, d.lat)
    path = ox.routing.shortest_path(G, a, b, weight="travel_time")
    if not path: raise ValueError("No drivable path between O and D inside the bbox")
    length = sum(G[u][v][0]["length"] for u, v in zip(path, path[1:]))
    tt = sum(G[u][v][0]["travel_time"] for u, v in zip(path, path[1:]))
    return {"length_m": length, "freeflow_s": tt}

# ---------- 2. DistanceMatrix.ai ----------
def snap_db():
    c = sqlite3.connect(DB)
    c.execute("create table if not exists s(ts text, key text, base real, traffic real)"); return c

def fetch_snapshot(o, d):
    """Free tier has NO historical data: call this on a schedule (e.g. /api/snapshot via cron/Task Scheduler)."""
    if not KEY: return None
    r = requests.get("https://api.distancematrix.ai/maps/api/distancematrix/json", timeout=30, params={
        "origins": f"{o.lat},{o.lon}", "destinations": f"{d.lat},{d.lon}", "departure_time": "now", "key": KEY})
    el = r.json()["rows"][0]["elements"][0]
    base, traf = el["duration"]["value"], el.get("duration_in_traffic", el["duration"])["value"]
    c = snap_db(); c.execute("insert into s values(?,?,?,?)", (dt.datetime.now().isoformat(),
        f"{o.lat:.4f},{o.lon:.4f}>{d.lat:.4f},{d.lon:.4f}", base, traf)); c.commit()
    return base, traf

def delay_ratio(o, d, days):
    key = f"{o.lat:.4f},{o.lon:.4f}>{d.lat:.4f},{d.lon:.4f}"
    df = pd.read_sql("select * from s where key=?", snap_db(), params=(key,), parse_dates=["ts"])
    df = df[df.ts > dt.datetime.now() - dt.timedelta(days=days)]
    if df.empty: return None
    df["ratio"] = (df.traffic / df.base).clip(1.0, 4.0); df["h"] = df.ts.dt.hour
    return df.groupby("h").ratio.mean()          # observed mean delay ratio per hour

# ---------- 3. Demand model ----------
def vc_from_ratio(r):             # invert BPR: t/t0 = 1 + 0.15 (v/c)^4
    return min(1.3, ((max(r, 1.0) - 1) / 0.15) ** 0.25)

def hourly_volumes(ratios, lanes, cap_per_lane=1500):    # PCU/h/lane ~ mixed-flow, urban
    cap = lanes * cap_per_lane
    if ratios is None or ratios.empty:
        vc = pd.Series({h: 0.6 * HOURLY[h] for h in range(24)})        # fallback: typical profile, peak v/c=0.6
    else:
        scale = np.mean([vc_from_ratio(r) / HOURLY[h] for h, r in ratios.items()])   # calibrate profile to observations
        obs = {h: vc_from_ratio(r) for h, r in ratios.items()}
        vc = pd.Series({h: obs.get(h, scale * HOURLY[h]) for h in range(24)})
    return (vc * cap).round().astype(int)         # veh/h (PCU-based, treated as veh/h for simplicity)

def poisson_trips(net, o, d, vols, job: Path):
    """Main O/D demand: non-homogeneous Poisson process (exponential headways) with mixed fleet."""
    rnd, rows = random.Random(42), []
    ex, ey = net.convertLonLat2XY(o.lon, o.lat); dx, dy = net.convertLonLat2XY(d.lon, d.lat)
    pick = lambda x, y: min(((e, dist) for e, dist in net.getNeighboringEdges(x, y, 300) if e.allows("passenger")),
                            key=lambda t: t[1], default=(None,))[0]
    eo, ed = pick(ex, ey), pick(dx, dy)
    if not eo or not ed: raise ValueError("O or D is >300 m from a drivable edge in the converted net")
    types, w = list(MIX), list(MIX.values()); i = 0
    for h, v in vols.items():
        t = h * 3600 + rnd.expovariate(max(v, 1) / 3600)
        while t < (h + 1) * 3600:
            rows.append((t, f"od{i}", rnd.choices(types, w)[0])); i += 1
            t += rnd.expovariate(max(v, 1) / 3600)
    with open(job / "od.trips.xml", "w") as f:
        f.write('<routes>\n')
        for t, vid, ty in rows:
            f.write(f'<trip id="{vid}" type="{ty}" depart="{t:.1f}" from="{eo.getID()}" to="{ed.getID()}" departLane="best" departSpeed="max"/>\n')
        f.write('</routes>')
    return len(rows)

# ---------- Pipeline ----------
def pipeline(jid: str, r: Req):
    job = WORK / jid; job.mkdir(exist_ok=True); st = JOBS[jid]
    def step(m): st["status"] = m
    try:
        step("Downloading OSM"); fetch_osm(r.bbox, job / "map.osm")
        step("Analysing corridor (OSMnx)"); info = corridor_info(r.bbox, r.origin, r.dest)
        step("Fetching traffic delay"); fetch_snapshot(r.origin, r.dest)
        ratios = delay_ratio(r.origin, r.dest, r.days)
        step("Converting to SUMO net")
        run(["netconvert", "--osm-files", "map.osm", "-o", "net.net.xml", "--geometry.remove", "--roundabouts.guess",
             "--ramps.guess", "--junctions.join", "--tls.guess-signals", "--tls.join", "--remove-edges.isolated",
             "--osm.sidewalks", "false", "--no-warnings"], job)
        net = sumolib.net.readNet(str(job / "net.net.xml"))
        step("Building demand")
        vols = hourly_volumes(ratios, r.lanes); (job / "vtypes.add.xml").write_text(VTYPES)
        n_od = poisson_trips(net, r.origin, r.dest, vols, job)
        total_bg = int(vols.sum() * r.background_share)         # Poisson-like background on minor roads
        run([sys.executable, TOOLS / "randomTrips.py", "-n", "net.net.xml", "-e", "86400", "-p", f"{86400/max(total_bg,1):.2f}",
             "--binomial", "4", "--prefix", "bg", "--fringe-factor", "5", "-o", "bg.trips.xml",
             "--trip-attributes", 'departLane="best" departSpeed="max"', "--validate"], job)
        step("Routing (duarouter)")
        run(["duarouter", "-n", "net.net.xml", "--additional-files", "vtypes.add.xml", "-r", "od.trips.xml,bg.trips.xml",
             "-o", "routes.rou.xml", "--ignore-errors", "--repair", "--no-warnings"], job)
        cfg_extra = ""
        if r.optimize_signals:
            step("Optimizing signals (Webster cycle adaptation)")
            try:
                run([sys.executable, TOOLS / "tlsCycleAdaptation.py", "-n", "net.net.xml", "-r", "routes.rou.xml",
                     "-b", "0", "-o", "tls.add.xml"], job); cfg_extra = ",tls.add.xml"
            except RuntimeError as e: st["warning"] = str(e)[:300]
        (job / "run.sumocfg").write_text(f"""<configuration>
<input><net-file value="net.net.xml"/><route-files value="routes.rou.xml"/><additional-files value="vtypes.add.xml{cfg_extra}"/></input>
<time><begin value="0"/><end value="86400"/></time>
<processing><lateral-resolution value="0.4"/><ignore-route-errors value="true"/><time-to-teleport value="300"/></processing>
<output><tripinfo-output value="tripinfo.xml"/><summary-output value="summary.xml"/></output></configuration>""")
        result = {"corridor": info, "od_vehicles_24h": n_od, "hourly_volume": vols.to_dict(),
                  "calibrated_with_snapshots": ratios is not None, "folder": str(job.resolve())}
        if r.simulate:
            step("Simulating"); run(["sumo", "-c", "run.sumocfg", "--no-warnings"], job)
            ti = pd.read_xml(job / "tripinfo.xml")
            result["sim"] = {"trips": len(ti), "mean_duration_s": round(ti.duration.mean(), 1),
                             "mean_waiting_s": round(ti.waitingTime.mean(), 1), "mean_time_loss_s": round(ti.timeLoss.mean(), 1)}
        st.update(status="done", result=result)
    except Exception as e:
        st.update(status="error", error=str(e))

@app.post("/api/run")
def start(r: Req, bg: BackgroundTasks):
    jid = uuid.uuid4().hex[:8]; JOBS[jid] = {"status": "queued"}; bg.add_task(pipeline, jid, r); return {"job": jid}

@app.get("/api/job/{jid}")
def job(jid: str):
    if jid not in JOBS: raise HTTPException(404)
    return JOBS[jid]

@app.post("/api/snapshot")
def snapshot(o: Pt, d: Pt):
    """Call every 15-30 min (Windows Task Scheduler/cron) to build your own historical baseline."""
    return {"base_s_traffic_s": fetch_snapshot(o, d)}

@app.get("/api/job/{jid}/file/{name}")
def get_file(jid: str, name: str):
    from fastapi.responses import FileResponse
    p = WORK / jid / Path(name).name
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p)
