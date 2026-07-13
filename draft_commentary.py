#!/usr/bin/env python3
# =====================================================================
#  Draft the monthly analyst note and open a GitHub issue for approval.
#  NOTHING is published by this script. The live page keeps showing the
#  last approved note until you approve the draft.
#
#  Secrets needed (GitHub repo > Settings > Secrets > Actions):
#    ANTHROPIC_API_KEY   - from console.anthropic.com
#    (GITHUB_TOKEN is provided automatically by Actions)
# =====================================================================
import os, json, sys, datetime as dt, urllib.request

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
GH_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()
REPO = os.environ.get("GITHUB_REPOSITORY", "").strip()   # e.g. marty21gt/fragility-monitor
if not API_KEY or not GH_TOKEN or not REPO:
    print("ERROR: missing ANTHROPIC_API_KEY / GITHUB_TOKEN / GITHUB_REPOSITORY"); sys.exit(1)

def log(m): print(m, flush=True)

# ---------- read the current reading from data.json ----------
with open("data.json", encoding="utf-8") as f:
    data = json.load(f)
cur = data.get("current", {})
V = cur.get("vulnerability", []); T = cur.get("trigger", [])
def blend(rows):
    # only SCORING gauges feed V/T -- context gauges (margin debt, NFCI) are display-only
    v = [r["frag"] for r in rows if r.get("frag") is not None and r.get("score", True)]
    if not v: return None
    s = sorted(v, reverse=True); tail = sum(s[:2])/2 if len(s) >= 2 else s[0]
    return 0.70*(sum(v)/len(v)) + 0.30*tail
vs, ts = blend(V), blend(T)
state = ("Danger" if (vs>=.62 and ts>=.55) else "Elevated" if vs>=.62
         else "Stress" if ts>=.55 else "Calm") if (vs and ts) else "Unknown"
gauges = "\n".join(f"  - {r['label']}: {r['frag']:.2f}  ({r['sub']})"
                   for r in V+T if r.get("score", True))
context = "\n".join(f"  - {r['label']}: {r['frag']:.2f}  ({r['sub']})"
                    for r in V+T if not r.get("score", True)) or "  (none)"
prior = ""
if os.path.exists("commentary.txt"):
    prior = open("commentary.txt", encoding="utf-8").read().strip()

PROMPT = f"""You are drafting the "Analyst note — what could light the spark" section of a
systemic-risk dashboard for a Registered Investment Advisor. It is educational content, not advice.

Current reading ({dt.date.today():%B %Y}):
  Vulnerability: {vs:.2f}   Trigger: {ts:.2f}   State: {state}
Scoring gauges (these produce V and T):
{gauges}

Context gauges (NOT in the score -- for qualitative colour only):
{context}

The framework: Vulnerability = how primed/leveraged/overvalued the system is (the "tinder").
Trigger = whether stress is actively igniting now (the "spark"). A large drawdown needs both.
High vulnerability alone is ambient context, NOT an alarm, and can persist for years.

Last month's note (for continuity — say what has CHANGED, don't just repeat it):
{prior[:1200]}

Write 4-6 sentences that:
  - state plainly what the gauges say and what the state means
  - identify the most plausible SPARK that the quantitative gauges cannot see
    (a qualitative risk: concentration, policy, credit chain, geopolitics, etc.)
  - note explicitly if nothing is igniting yet

Hard rules — this is published by a regulated advisor:
  - NO predictions or forecasts ("will fall", "expect a crash"). Describe conditions, not outcomes.
  - NO investment recommendations or advice to buy/sell anything.
  - NO performance claims or return figures.
  - Measured, factual, non-alarmist tone. Plain prose, no headings, no bullets.
Return ONLY the note text."""

req = urllib.request.Request(
    "https://api.anthropic.com/v1/messages",
    data=json.dumps({"model":"claude-sonnet-4-6","max_tokens":700,
                     "messages":[{"role":"user","content":PROMPT}]}).encode(),
    headers={"content-type":"application/json","x-api-key":API_KEY,
             "anthropic-version":"2023-06-01"})
with urllib.request.urlopen(req, timeout=90) as r:
    resp = json.load(r)
draft = "".join(b.get("text","") for b in resp.get("content",[]) if b.get("type")=="text").strip()
if not draft:
    log("ERROR: empty draft from API"); sys.exit(1)
log("Draft generated.\n" + draft)

# ---------- open a GitHub issue for review ----------
month = f"{dt.date.today():%B %Y}"
body = f"""**Draft analyst note for {month}** — review before it goes live.

Current reading: **Vulnerability {vs:.2f} · Trigger {ts:.2f} · {state}**

---

{draft}

---

### To APPROVE and publish
Edit `commentary.txt` in the repo, paste the text above (edit it however you like),
and commit. The live page picks it up on the next data refresh — or run the
**Update Fragility Data** workflow to publish immediately.

### To REJECT
Just close this issue. The page keeps showing the last approved note; nothing changes.

<sub>Drafted automatically. Nothing is published without your action.</sub>
"""
payload = json.dumps({"title": f"Analyst note for review — {month}",
                      "body": body, "labels": ["commentary-review"]}).encode()
ireq = urllib.request.Request(f"https://api.github.com/repos/{REPO}/issues", data=payload,
    headers={"Authorization": f"Bearer {GH_TOKEN}", "Accept":"application/vnd.github+json",
             "Content-Type":"application/json", "User-Agent":"fragility-bot"})
with urllib.request.urlopen(ireq, timeout=45) as r:
    issue = json.load(r)
log(f"Opened issue #{issue['number']} for review: {issue['html_url']}")

# archive the draft for the compliance record
os.makedirs("commentary_archive", exist_ok=True)
with open(f"commentary_archive/{dt.date.today():%Y-%m}-draft.txt","w",encoding="utf-8") as f:
    f.write(f"DRAFTED {dt.date.today():%Y-%m-%d} | V={vs:.2f} T={ts:.2f} {state}\n\n{draft}\n")
log("Archived draft.")
