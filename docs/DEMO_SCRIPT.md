# Live E2E Demo Script — Landslide Early Warning System

Literal, step-by-step script for a judging demonstration. Every API call below was
executed and verified against the live stack (PostgreSQL 17 + Memurai/Redis + Django +
Celery + Vite). Copy/paste each line into **PowerShell** and compare the output to the
"Expected" lines.

---

## 0. Model status — read this first (honesty requirement)

**This demo runs the published threshold model ONLY. No trained ML classifier artifact
exists in the repository.**

- The system's documented architecture makes the threshold model the **always-active
  baseline** and the ML classifier an *optional* Phase-2 refinement layer on top
  (see `backend/apps/ml_bridge/ml/threshold_model.py` docstring).
- `recompute_risk` evaluates the latest stored weather readings against the published
  NE-Himalaya rainfall threshold **E(mm) = -11.10 + 0.62 × D(hr)** and persists the
  resulting level back onto the RiskZone.
- The dataset-builder for ML training (`apps.ml_bridge.ingestion.pipeline`) exists but is
  not wired into the operational recompute path.
- A `*.pkl/.joblib` model file was never generated/trained and **must not be claimed** in
  the pitch. If a judge asks "where is the trained model?", the correct answer is:
  *"Rainfall-trigger thresholds are the always-active baseline; the ML classifier is the
  Phase-2 refinement trained from the assembled NER dataset — not yet shipped."*

**Demo maths for the script:** reading = 87.6 mm over the 48 h operational window.
Threshold: 0.62×48 − 11.10 = **18.66 mm**. 87.6 > 18.66 → exceedance, margin **+68.94 mm**
→ displayed as "18.7mm threshold, 68.9mm above".

---

## 1. Infrastructure checks (2 tables)

```powershell
Get-Service postgresql-X64-17, Memurai | Select-Object Name, Status
python -c "import redis; print(redis.Redis(host='127.0.0.1').ping())"
```

Expected:

```
postgresql-X64-17  Running
Memurai            Running
True
```

## 2. Start the stack (three terminals, all in repo root)

Terminal A — backend:

```powershell
cd backend
python manage.py runserver 0.0.0.0:8000 --noreload
```

Terminal B — Celery worker. **Windows REQUIRES `--pool=solo`** (the default prefork pool
crashes with `ValueError: not enough values to unpack (expected 3, got 0)` — a known
Windows/billiard issue, not a code bug; production/Linux keeps prefork):

```powershell
cd backend
python -m celery -A config worker --loglevel=info --pool=solo
```

Terminal C — frontend (Vite dev server):

```powershell
cd frontend-dashboard
npm run dev
```

Expected: Vite prints `Local: http://localhost:5173/`. Also verify the port owner is the
backend you just started (stale processes on :8000 have bitten us before):

```powershell
(Get-NetTCPConnection -LocalPort 8000 -State Listen).OwningProcess
```

Optional — Celery beat with the demo 20/30/60 s schedule (fires the real tasks through the
broker to the worker above; see Appendix A):

```powershell
cd backend
$env:DJANGO_SETTINGS_MODULE = 'config.settings_demo'
python -m celery -A config beat --loglevel=info
```

## 3. Seed the demo dataset (idempotent — safe to re-run anytime)

```powershell
cd backend
python manage.py demo_seed
```

Expected (last line): `Demo dataset ready: 6 zones | 2 alerts | 2 reports | 1 readings |
user 9876543210 (state_admin)` and `Test zone 'Kalimpong-Darjeeling Foothills' at baseline
Low ... recompute_risk will elevate it to High.`

The seed creates:
- state_admin user (phone `9876543210`)
- 5 base NER zones + 2 alerts + 2 field reports
- the **Kalimpong-Darjeeling Foothills** test zone at baseline **Low**, with one fresh AWS
  reading of **87.6 mm** (station `KLP-AWS-001`, 30 min old)

> Re-running `demo_seed` resets zones and the test reading. Dispatched alerts from a
> previous run persist — clear the test one with:
> `python manage.py shell -c "from apps.alerts.models import Alert; Alert.objects.filter(zone_id=7).delete()"`

## 4. Login and capture the JWT

```powershell
$login = curl.exe -s -X POST http://localhost:8000/api/v1/auth/login -H "Content-Type: application/json" -d '{"phone_number":"9876543210","otp":"test-otp"}'
$TOKEN = ($login | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
echo $TOKEN.Substring(0,20)
```

Expected: a ~20-char JWT prefix. **The token field is `access_token`** (not `token`).

## 5. Located the test zone id (idempotent)

```powershell
$ZONE = (curl.exe -s http://localhost:8000/api/v1/risk-zones/ -H "Authorization: Bearer $TOKEN" | python -c "import sys,json; print([z['id'] for z in json.load(sys.stdin)['results'] if z['zone_name']=='Kalimpong-Darjeeling Foothills'][0])")
echo $ZONE
```

Expected: `7` (or whatever id exists in your DB — use it everywhere below).

## 6. BEFORE recompute — zone is Low (stale baseline)

```powershell
curl.exe -s http://localhost:8000/api/v1/risk-zones/$ZONE/explanation/ -H "Authorization: Bearer $TOKEN"
curl.exe -s http://localhost:8000/api/v1/dashboard/summary -H "Authorization: Bearer $TOKEN"
```

Expected explanation payload:

```json
{ "zone_id": 7, "zone_name": "Kalimpong-Darjeeling Foothills",
  "risk_level": "Low",
  "explanation": "2-day cumulative rainfall of 87.6mm exceeds the NE Himalaya threshold of 18.7mm for this duration (68.9mm above threshold).", ... }
```

Note the nuance for the narrative: the explanation panel is computed **live from
readings**, so the text already shows the exceedance — but the persisted zone level
(`risk_level` in the payload) is still **Low** because the scheduler job has not
recomputed the zone yet. Dashboard counts: `High: 1`.

## 7. TRIGGER — recompute risk (management command, deterministic)

```powershell
python manage.py recompute_risk
```

Expected (tail): `"zone_id": 7`, `"updated": true`, `"risk_level": "High"`,
`"explanation": "2-day cumulative rainfall of 87.6mm exceeds..."`. The other 5 zones report
`"updated": false` with reason "no weather readings with rainfall for this zone".

**The Celery route (same result, through broker + worker):** with Terminals B running,

```powershell
python -m celery -A config call apps.ml_bridge.tasks.recompute_risk
```

or let beat fire `recompute-risk-demo-60s` (Appendix A) and watch the worker log:
`Task apps.ml_bridge.tasks.recompute_risk[...] succeeded`.

## 8. AFTER recompute — level is now High, persisted, referenced by the reading

```powershell
curl.exe -s http://localhost:8000/api/v1/risk-zones/$ZONE/explanation/ -H "Authorization: Bearer $TOKEN"
curl.exe -s http://localhost:8000/api/v1/risk-zones/$ZONE/ -H "Authorization: Bearer $TOKEN"
curl.exe -s http://localhost:8000/api/v1/dashboard/summary -H "Authorization: Bearer $TOKEN"
```

Expected: explanation payload now has `"risk_level": "High"` with the same 87.6 mm-referencing
text; zone record shows `"current_risk_level": "High", "last_computed_at": "<now>"`;
summary shows `"High": 2` (Kalimpong promoted), `"Low": 1`.

## 9. Dispatch the alert (admin override via API, explanation text attached)

Write the payload to a file (avoids PowerShell JSON-quoting pain):

```powershell
@'
{
  "zone_id": 7,
  "risk_level": "High",
  "message": "Heavy rainfall past 48h in Kalimpong-Darjeeling foothills; slope saturation heightened. Move residents near vulnerable cut slopes to safer ground.",
  "channel": "sms",
  "language": "en",
  "explanation": "2-day cumulative rainfall of 87.6mm exceeds the NE Himalaya threshold of 18.7mm for this duration (68.9mm above threshold)."
}
'@ | Set-Content "$env:TEMP\dispatch_body.json"

curl.exe -s -X POST http://localhost:8000/api/v1/alerts/dispatch/ -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d "@$env:TEMP\dispatch_body.json"
```

Expected: `{"status":"dispatched","alert_id":3}`. Then confirm it is listed with the
explanation populated:

```powershell
curl.exe -s http://localhost:8000/api/v1/alerts/ -H "Authorization: Bearer $TOKEN"
```

Expected: newest alert first — `id 3`, `risk_level "High"`, `zone_name
"Kalimpong-Darjeeling Foothills"`, `channel "sms"`, `dispatched_at` set, and `explanation`
containing `87.6mm` / `18.7mm`.

> Guard note: `dispatch_alert` requires a staff/admin role (state_admin ✓) and the rules
> engine refuses dispatch below `High` (`should_dispatch_alert`).

## 10. FRONTEND — the dashboard reflects everything via the Refresh button

1. Open **http://localhost:5173** in Chrome.
2. Login: phone `9876543210`, OTP `test-otp`, click **Sign in**.
3. **Map tab:** the NER risk zones render as colored polygons (Shillong red, Kalimpong
   **green**). Summary cards show `6 zones · risk distribution · 2 active alerts ·
   1 pending report`.
4. Click the **Kalimpong-Darjeeling Foothills** polygon. Zone panel (right) shows badge
   **Risk level: Low** (green) while the explanation block beneath already displays the
   threshold text with the 87.6 mm / 18.7 mm numbers, plus the "Thresholds checked" JSON
   and the readings table.
5. Run the recompute step (§7) in the terminal.
6. Back in the browser click **Refresh data** (top-right header). WITHOUT reloading the
   page: the badge flips to **Risk level: High** (orange), the map polygon recolors,
   and the summary card shows `High: 2`. Explanation text is unchanged (same live
   reading-driven text).
7. Click the **Alerts** tab: the new **High · Kalimpong-Darjeeling Foothills · Channel:
   sms** card is on top with the explanation paragraph.
8. Bonus: the **Reports** tab shows the two seeded field reports (1 pending).

Token lifetime is in-memory only — a hard F5 logs you out; re-login via the same OTP
rather than reloading, or just use *Refresh data* (that's the point of the button).

---

## Verification checklist (everything above was executed successfully)

| Step | Live result |
|---|---|
| Infra (Postgres + Redis) | `Running` / `True` |
| runserver + worker (--pool=solo) + vite | 200 OK app, worker ready |
| demo_seed | 6 zones / 2 alerts / 2 reports / 1 reading |
| Login | JWT via `access_token` |
| Explanation BEFORE | `risk_level: Low`, text cites 87.6 vs 18.7 mm |
| recompute_risk | zone 7 → `High`, `last_computed_at` set |
| Explanation AFTER | `risk_level: High`, `last_computed_at` non-null |
| Summary AFTER | `High: 2`, `Low: 1`, `active_alerts: 3` |
| Alert dispatch | 201 `{"status":"dispatched","alert_id":3}` |
| GET /alerts/ | new alert first, explanation populated |
| CORS from :5173 | `access-control-allow-origin: http://localhost:5173` |
| Frontend build | `tsc && vite build` clean (160 modules) |
| Backend tests | 164 passed `python -m pytest tests -q` |

---

## Known platform quirks (memorize)

1. **Celery worker on Windows** — always `--pool=solo` (prefork crashes on billiard).
2. **Login token key** — `access_token` (Bearer header).
3. **`/api/v1/dashboard/summary` has NO trailing slash** (plain `path()`). DRF router
   endpoints (`/risk-zones/`, `/alerts/`, `/reports/`, `/explanation/`, `/dispatch/`)
   DO require it (APPEND_SLASH redirect otherwise).
4. **`weather/forecast`** (plain path): `GET /api/v1/weather/forecast?zone_id=7`.
5. Local DB is plain PostgreSQL (no GDAL): `GIS_AVAILABLE=False`, geometry columns are
   JSON; PostGIS/GDAL geometry is the **container/production** path.

---

## Appendix A — Celery beat, live (advanced judging attention)

With the worker running (§2) and beat started with `config.settings_demo`:

- `ingest-rainfall-demo-20s` → fires every 20 s (IMD/NASA fetch, degrades gracefully
  offline to `{"status":"error"}` result — still an executed task)
- `ingest-soil-moisture-demo-30s` → every 30 s (returns 0 records live)
- `recompute-risk-demo-60s` → every 60 s → **persists zone 7 to High** exactly like §7

Watch the worker terminal for `Task ... succeeded` lines and `Risk recomputation
completed: 1 zones updated`.

## Appendix B — fresh-machine bring-up

```powershell
# once — create DB and grant (assumes Postgres superuser password is 'postgres')
psql -h 127.0.0.1 -U postgres -c "CREATE DATABASE landslide_ews;"
cd backend
python manage.py migrate
python manage.py demo_seed
```

---

## Fallback plan (live failure safety)

**A recorded fallback video MUST exist** covering §6→§10 (before-explanation, recompute,
after-explanation, alert dispatch, Refresh-button UI update). If any live step fails
during judging:

1. Pause, point to the video, and continue from the next step that works.
2. `python manage.py demo_seed` re-seeds everything (idempotent) — a failed §6/§7 can be
   re-run live, zero cleanup needed for zones/readings.
3. If the frontend misbehaves, the entire chain is still provable from the terminal
   (all REST verification is API-only); present the JSON payloads directly.