"""OSM -> SUMO pipeline. Run: uvicorn main:app --reload --port 8000"""
import warnings, os, sys, uuid, random, sqlite3, subprocess, shutil, datetime as dt
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, pandas as pd, requests, osmnx as ox
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()
KEY = os.getenv("DISTANCEMATRIX_KEY", "")
WORK, DB = Path("jobs"), "snapshots.db"
WORK.mkdir(exist_ok=True)
SUMO_HOME = os.environ.get("SUMO_HOME")
if not SUMO_HOME:
    import sumo; SUMO_HOME = sumo.SUMO_HOME          # pip eclipse-sumo
TOOLS = Path(SUMO_HOME) / "tools"
os.environ["SUMO_HOME"] = str(SUMO_HOME)
os.environ["PATH"] = str(Path(SUMO_HOME) / "bin") + os.pathsep + os.environ["PATH"]
sys.path.append(str(TOOLS))
import sumolib

app = FastAPI(title="OSM-SUMO Bridge")
app.add_middleware(CORSMiddleware, allow_origins=[],
                   allow_origin_regex=r"https://.*\.github\.io|http://(localhost|127\.0\.0\.1)(:\d+)?",
                   allow_methods=["*"], allow_headers=["*"])
JOBS: dict = {}

@app.middleware("http")          # Chrome "Local Network Access": public https page -> localhost
async def pna(request, call_next):
    resp = await call_next(request)
    resp.headers["Access-Control-Allow-Private-Network"] = "true"
    return resp

class Pt(BaseModel): lat: float; lon: float
class Req(BaseModel):
    bbox: list[float] = Field(..., min_length=4, max_length=4, description="[south, west, north, east]")
    origins: list[Pt]; dests: list[Pt]
    lanes: int = Field(2, ge=1, le=8)
    days: int = Field(7, ge=1, le=60)
    optimize_signals: bool = True
    background_share: float = Field(0.35, ge=0, le=3)
    start_hour: int = Field(0, ge=0, le=23)     # simulate only this window (e.g. 7-10 for the AM peak)
    end_hour: int = Field(24, ge=1, le=24)
    simulate: bool = True

# ---------- South Asian mixed-flow assumptions (tune with local counts!) ----------
HOURLY = np.array([.18,.12,.10,.10,.15,.30,.55,.80,1.0,.90,.80,.80,.85,.85,.80,.82,.90,1.0,.95,.80,.60,.45,.32,.24])
MIX = {"car":.30,"motorcycle":.35,"cng":.15,"bus":.08,"truck":.05,"rickshaw":.07}
VTYPES = """<routes>
<vType id="car" vClass="passenger" length="4.2" width="1.8" maxSpeed="16" accel="2.5" sigma="0.6" latAlignment="center" minGapLat="0.3"/>
<vType id="motorcycle" vClass="motorcycle" length="2.0" width="0.8" maxSpeed="17" accel="3.5" sigma="0.7" latAlignment="arbitrary" minGapLat="0.15" lcPushy="0.8"/>
<vType id="cng" vClass="passenger" length="2.6" width="1.4" maxSpeed="12" accel="2" sigma="0.7" latAlignment="arbitrary" minGapLat="0.2"/>
<vType id="bus" vClass="bus" length="11" width="2.5" maxSpeed="13" accel="1.2" sigma="0.5" latAlignment="right"/>
<vType id="truck" vClass="truck" length="8" width="2.5" maxSpeed="12" accel="1.0" sigma="0.5" latAlignment="right"/>
<vType id="rickshaw" vClass="bicycle" length="2.5" width="1.0" maxSpeed="4" accel="0.8" sigma="0.8" latAlignment="right"/>
</routes>"""

def run(cmd, cwd):
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if p.returncode: raise RuntimeError(f"{Path(str(cmd[0])).name} failed:\n{(p.stderr or p.stdout)[-1500:]}")
    return p.stdout

# ---------- 1. OSM ----------
MIRRORS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
           "https://overpass.private.coffee/api/interpreter"]
def fetch_osm(bbox, out: Path):
    s, w, n, e = bbox
    q = (f'[out:xml][timeout:180];(way["highway"]["highway"!~"footway|path|steps|cycleway|pedestrian|bridleway|'
         f'corridor|proposed|construction|elevator|platform"]["access"!~"private|no"]({s},{w},{n},{e});>;);out meta;')
    err = "no response"
    for url in MIRRORS:
        try:
            r = requests.post(url, data={"data": q}, headers={"User-Agent": "OSMSUMOBridge/1.0 (research)"}, timeout=200)
            if r.status_code == 200 and b"<osm" in r.content[:2000] and b"<way" in r.content:
                out.write_bytes(r.content); return
            err = f"{url}: HTTP {r.status_code}" if r.status_code != 200 else f"{url}: no roads in this area"
        except requests.RequestException as ex:
            err = f"{url}: {ex}"
    raise RuntimeError(f"All Overpass mirrors failed (last: {err}). Select a smaller area and retry.")

HWY_KPH = {"motorway": 80, "motorway_link": 50, "trunk": 60, "trunk_link": 40, "primary": 50, "primary_link": 35,
           "secondary": 40, "secondary_link": 30, "tertiary": 30, "tertiary_link": 25, "unclassified": 25,
           "residential": 20, "living_street": 10, "service": 15, "road": 20}
def build_graph(osm_file):
    """OSM XML -> routing graph with speeds (OSM in Bangladesh rarely has maxspeed, so use highway-type defaults)."""
    G = ox.graph_from_xml(str(osm_file))
    G = ox.add_edge_speeds(G, hwy_speeds=HWY_KPH, fallback=20)
    return ox.truncate.largest_component(ox.add_edge_travel_times(G), strongly=True)

def corridor_info(G, o, d):
    """Shortest path on the OSM graph -> length (m) and free-flow time (s)."""
    a, b = ox.nearest_nodes(G, o.lon, o.lat), ox.nearest_nodes(G, d.lon, d.lat)
    path = ox.routing.shortest_path(G, a, b, weight="travel_time") if a != b else None
    if not path: raise ValueError("No drivable path between this O and D inside the area (enlarge it or move the points)")
    ed = [min(G[u][v].values(), key=lambda x: x["travel_time"]) for u, v in zip(path, path[1:])]
    return {"length_m": round(sum(x["length"] for x in ed)), "freeflow_s": round(sum(x["travel_time"] for x in ed))}

# ---------- 2. DistanceMatrix.ai ----------
def db():
    c = sqlite3.connect(DB)
    c.execute("create table if not exists s(ts text, key text, base real, traffic real)"); return c
def okey(o, d): return f"{o.lat:.4f},{o.lon:.4f}>{d.lat:.4f},{d.lon:.4f}"

def fetch_snapshot(o, d):
    """Free tier has NO history: call on a schedule (/api/snapshot via cron/Task Scheduler) to build your own."""
    if not KEY: raise RuntimeError("DISTANCEMATRIX_KEY missing in .env")
    r = requests.get("https://api.distancematrix.ai/maps/api/distancematrix/json", timeout=30, params={
        "origins": f"{o.lat},{o.lon}", "destinations": f"{d.lat},{d.lon}", "departure_time": "now", "key": KEY})
    r.raise_for_status(); j = r.json()
    el = j["rows"][0]["elements"][0]
    if el.get("status", "OK") != "OK": raise RuntimeError(f"DistanceMatrix.ai: {el.get('status')} {j.get('error_message', '')}")
    base, traf = el["duration"]["value"], el.get("duration_in_traffic", el["duration"])["value"]
    c = db()
    try: c.execute("insert into s values(?,?,?,?)", (dt.datetime.now().isoformat(), okey(o, d), base, traf)); c.commit()
    finally: c.close()
    return base, traf

def delay_ratio(o, d, days):
    c = db()
    try: df = pd.read_sql("select * from s where key=?", c, params=(okey(o, d),), parse_dates=["ts"])
    finally: c.close()
    df = df[df.ts > dt.datetime.now() - dt.timedelta(days=days)]
    if df.empty: return None
    df["ratio"] = (df.traffic / df.base).clip(1.0, 4.0); df["h"] = df.ts.dt.hour
    return df.groupby("h").ratio.mean()          # observed mean delay ratio per hour

# ---------- 3. Demand model ----------
def vc_from_ratio(r):             # invert BPR: t/t0 = 1 + 0.15 (v/c)^4
    return min(1.3, ((max(r, 1.0) - 1) / 0.15) ** 0.25)

def hourly_volumes(ratios, lanes, cap_per_lane=1500):    # veh/h/lane, mixed-flow urban
    cap = lanes * cap_per_lane
    if ratios is None or ratios.empty:
        vc = pd.Series({h: 0.6 * HOURLY[h] for h in range(24)})        # fallback: typical profile, peak v/c=0.6
    else:
        good = {h: vc_from_ratio(r) for h, r in ratios.items() if r >= 1.1}   # BPR inversion only valid when delayed
        scale = float(np.mean([v / HOURLY[h] for h, v in good.items()])) if good else 0.6
        vc = pd.Series({h: good[h] if h in good else (min(scale * HOURLY[h], 0.3) if h in ratios.index else scale * HOURLY[h])
                        for h in range(24)})
    return (vc * cap).round().astype(int)

def pick_edges(net, o, d):
    """Closest drivable edges that are actually connected (tries 5x5 candidates)."""
    def cands(p):
        x, y = net.convertLonLat2XY(p.lon, p.lat)
        c = sorted(((e, k) for e, k in net.getNeighboringEdges(x, y, 300) if e.allows("passenger")), key=lambda t: t[1])
        return [e for e, _ in c[:5]]
    co, cd = cands(o), cands(d)
    if not co or not cd: raise ValueError("O or D is >300 m from a drivable edge in the converted net")
    for eo in co:
        for ed in cd:
            if eo != ed and net.getShortestPath(eo, ed, vClass="passenger")[0]: return eo, ed
    raise ValueError("O and D snap to edges with no connecting route in the converted net")

def poisson_trips(net, o, d, vols, job: Path, tag, h0, h1):
    """Main O/D demand: non-homogeneous Poisson process with mixed fleet."""
    rnd, rows = random.Random(tag), []
    eo, ed = pick_edges(net, o, d)
    types, w = list(MIX), list(MIX.values())
    for h in range(h0, h1):
        rate = max(int(vols[h]), 1) / 3600; t = h * 3600 + rnd.expovariate(rate)
        while t < (h + 1) * 3600:
            rows.append((t, rnd.choices(types, w)[0])); t += rnd.expovariate(rate)
    with open(job / f"{tag}.trips.xml", "w") as f:
        f.write('<routes>\n')
        for i, (t, ty) in enumerate(rows):
            f.write(f'<trip id="{tag}_{i}" type="{ty}" depart="{t:.1f}" from="{eo.getID()}" to="{ed.getID()}" departLane="best" departSpeed="max"/>\n')
        f.write('</routes>')
    return len(rows)

def background(job: Path, n, h0, h1, seed=42):
    """Random minor-road trips (randomTrips.py), re-timed to the hourly profile and given the mixed fleet."""
    span = (h1 - h0) * 3600
    run([sys.executable, TOOLS / "randomTrips.py", "-n", "net.net.xml", "-b", h0 * 3600, "-e", h1 * 3600,
         "-p", f"{span / (n * 1.3):.4f}", "--prefix", "bg", "--fringe-factor", "5", "--min-distance", "300",
         "--seed", seed, "-o", "bg.raw.xml", "--validate"], job)
    trips = ET.parse(job / "bg.raw.xml").getroot().findall("trip")[:n]
    if not trips: raise RuntimeError("randomTrips produced no valid trips")
    rnd = random.Random(seed); hrs = list(range(h0, h1)); wt = [HOURLY[h] for h in hrs]
    times = sorted(rnd.choices(hrs, wt)[0] * 3600 + rnd.random() * 3600 for _ in trips)
    root = ET.Element("routes")
    for t, tr in zip(times, trips):
        tr.set("depart", f"{t:.1f}"); tr.set("type", rnd.choices(list(MIX), list(MIX.values()))[0]); root.append(tr)
    ET.ElementTree(root).write(job / "bg.trips.xml")

def tls_programs(job: Path):
    """Rename Webster programs and activate them from t=0 with WAUTs (otherwise SUMO keeps the net's default plan)."""
    tree = ET.parse(job / "tls.add.xml"); root = tree.getroot()
    ids = sorted({t.get("id") for t in root.iter("tlLogic")})
    if not ids: raise RuntimeError("tlsCycleAdaptation produced no signal programs")
    for t in root.iter("tlLogic"): t.set("programID", "webster")
    for i in ids: ET.SubElement(root, "WAUT", refTime="0", id=f"w_{i}", startProg="webster")
    for i in ids: ET.SubElement(root, "wautJunction", wautID=f"w_{i}", junctionID=i)
    tree.write(job / "tls.add.xml")

def sim_stats(path: Path):
    rows = [dict(e.attrib) for _, e in ET.iterparse(path) if e.tag == "tripinfo"]
    if not rows: return {"trips": 0}
    df = pd.DataFrame(rows)
    for c in ("duration", "waitingTime", "timeLoss", "routeLength", "departDelay"): df[c] = df[c].astype(float)
    f = lambda x: {"trips": len(x), "mean_duration_s": round(x.duration.mean(), 1), "mean_waiting_s": round(x.waitingTime.mean(), 1),
                   "mean_time_loss_s": round(x.timeLoss.mean(), 1), "mean_depart_delay_s": round(x.departDelay.mean(), 1),
                   "mean_speed_kmh": round(x.routeLength.sum() / max(x.duration.sum(), 1) * 3.6, 1)}
    out = f(df); df["pair"] = df.id.str.extract(r"^(O\d+D\d+)_")[0]
    out["by_pair"] = {k: f(g) for k, g in df.dropna(subset=["pair"]).groupby("pair")}
    return out

# ---------- Pipeline ----------
def check(r: Req):
    if not r.origins or not r.dests: raise ValueError("Set at least one origin and one destination")
    if len(r.origins) * len(r.dests) > 25: raise ValueError("Max 25 O-D pairs per run (API quota)")
    if r.end_hour <= r.start_hour: raise ValueError("end_hour must be greater than start_hour")
    s, w, n, e = r.bbox
    if not (s < n and w < e): raise ValueError("Invalid area")
    km2 = (n - s) * 111.32 * (e - w) * 111.32 * np.cos(np.radians((n + s) / 2))
    if km2 > 25: raise ValueError(f"Area is {km2:.1f} km2 (max 25). Select a smaller area.")

def pipeline(jid: str, r: Req):
    job = WORK / jid; job.mkdir(exist_ok=True); st = JOBS[jid]
    def step(m): st["status"] = m
    try:
        check(r); h0, h1 = r.start_hour, r.end_hour
        pairs = [(i, j, o, d) for i, o in enumerate(r.origins, 1) for j, d in enumerate(r.dests, 1)]
        step("Downloading OSM (Overpass)"); fetch_osm(r.bbox, job / "map.osm")
        step("Converting to SUMO net")
        run(["netconvert", "--osm-files", "map.osm", "-o", "net.net.xml", "--proj.utm", "--geometry.remove",
             "--roundabouts.guess", "--ramps.guess", "--junctions.join", "--tls.guess-signals", "--tls.join",
             "--remove-edges.isolated", "--keep-edges.by-vclass", "passenger", "--no-warnings"], job)
        net = sumolib.net.readNet(str(job / "net.net.xml"))
        G = build_graph(job / "map.osm")
        (job / "vtypes.rou.xml").write_text(VTYPES)
        files, summary, bg_total, warns = [], [], 0, []
        for i, j, o, d in pairs:
            tag = f"O{i}D{j}"; step(f"Pair {tag}: delay + demand")
            try: fetch_snapshot(o, d)
            except Exception as ex: warns.append(f"{tag}: live snapshot skipped ({str(ex)[:120]})")
            ratios = delay_ratio(o, d, r.days)
            vols = hourly_volumes(ratios, r.lanes)
            n_od = poisson_trips(net, o, d, vols, job, tag, h0, h1)
            files.append(f"{tag}.trips.xml"); bg_total += int(sum(vols[h] for h in range(h0, h1)) * r.background_share)
            summary.append({"pair": tag, **corridor_info(G, o, d), "veh_in_window": n_od,
                            "peak_veh_h": int(vols[h0:h1].max()), "calibrated_with_snapshots": ratios is not None})
        if bg_total > 0:
            step("Background traffic (minor roads)")
            try: background(job, bg_total, h0, h1); files.append("bg.trips.xml")
            except Exception as ex: warns.append(f"background traffic skipped ({str(ex)[:200]})")
        step("Routing (duarouter)")
        run(["duarouter", "-n", "net.net.xml", "-r", ",".join(["vtypes.rou.xml"] + files), "-o", "routes.rou.xml",
             "--ignore-errors", "--repair", "--no-warnings"], job)
        with open(job / "routes.rou.xml") as f: has_vt = '<vType id="car"' in f.read(20000)
        extra = [] if has_vt else ["vtypes.rou.xml"]
        if r.optimize_signals:
            step("Optimizing signals (Webster cycle adaptation)")
            try:
                run([sys.executable, TOOLS / "tlsCycleAdaptation.py", "-n", "net.net.xml", "-r", "routes.rou.xml",
                     "-b", h0 * 3600, "-o", "tls.add.xml"], job)
                tls_programs(job); extra.append("tls.add.xml")
            except Exception as ex: warns.append(f"signal optimisation skipped ({str(ex)[:200]})")
        add = f'<additional-files value="{",".join(extra)}"/>' if extra else ""
        (job / "run.sumocfg").write_text(f"""<configuration>
<input><net-file value="net.net.xml"/><route-files value="routes.rou.xml"/>{add}</input>
<time><begin value="{h0 * 3600}"/><end value="{h1 * 3600 + 1800}"/></time>
<processing><lateral-resolution value="0.4"/><ignore-route-errors value="true"/><time-to-teleport value="300"/></processing>
<output><tripinfo-output value="tripinfo.xml"/><summary-output value="summary.xml"/></output></configuration>""")
        result = {"pairs": summary, "folder": str(job.resolve())}
        if r.simulate:
            step("Simulating"); run(["sumo", "-c", "run.sumocfg", "--no-warnings", "--no-step-log"], job)
            result["sim"] = sim_stats(job / "tripinfo.xml")
        st.update(status="done", result=result, warning="; ".join(warns) or None)
    except Exception as ex:
        st.update(status="error", error=str(ex))

@app.post("/api/run")
def start(r: Req, bg: BackgroundTasks):
    try: check(r)
    except ValueError as ex: raise HTTPException(400, str(ex))
    jid = uuid.uuid4().hex[:8]; JOBS[jid] = {"status": "queued"}; bg.add_task(pipeline, jid, r); return {"job": jid}

@app.get("/api/job/{jid}")
def job(jid: str):
    if jid not in JOBS: raise HTTPException(404, "Unknown job (server restarted?)")
    return JOBS[jid]

@app.post("/api/snapshot")
def snapshot(o: Pt, d: Pt):
    """Call every 15-30 min (Task Scheduler/cron) to build your own historical baseline."""
    try: base, traf = fetch_snapshot(o, d)
    except Exception as ex: raise HTTPException(502, str(ex))
    return {"base_s": base, "traffic_s": traf, "ratio": round(traf / base, 3)}

@app.get("/api/job/{jid}/file/{name}")
def get_file(jid: str, name: str):
    p = WORK / Path(jid).name / Path(name).name
    if not p.is_file(): raise HTTPException(404)
    return FileResponse(p)

@app.get("/api/job/{jid}/zip")
def get_zip(jid: str):
    d = WORK / Path(jid).name
    if not d.is_dir(): raise HTTPException(404)
    return FileResponse(shutil.make_archive(str(WORK / f"{d.name}_export"), "zip", d), filename=f"sumo_{d.name}.zip")
