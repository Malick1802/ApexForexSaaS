"""
Golden Signal Fleet Status Report Generator
=============================================
Scans models/specialist for all certified models and generates
a full-metrics Markdown report to the Antigravity artifact.

Usage:
    python scripts/generate_training_report.py
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "specialist"
LOG_FILE = PROJECT_ROOT / "logs" / "specialist_progressive.log"
ARTIFACT_PATH = Path(r"C:\Users\artem\.gemini\antigravity\brain\adf666f6-f704-4159-a6e4-40b10231b3ba\training_fleet_status.md")


def scan_models():
    """Scan all specialist model directories for config.json metrics."""
    rows = []
    if not MODELS_DIR.exists():
        return rows

    for pair_dir in sorted(MODELS_DIR.iterdir()):
        if not pair_dir.is_dir():
            continue
        pair = pair_dir.name
        buy_wr, buy_tr = "—", "—"
        sell_wr, sell_tr = "—", "—"

        buy_conf = pair_dir / "BUY" / "config.json"
        sell_conf = pair_dir / "SELL" / "config.json"

        if buy_conf.exists():
            try:
                with open(buy_conf) as f:
                    c = json.load(f)
                buy_wr = round(c.get("win_rate", 0) * 100, 1)
                buy_tr = c.get("trades", 0)
            except Exception:
                pass

        if sell_conf.exists():
            try:
                with open(sell_conf) as f:
                    c = json.load(f)
                sell_wr = round(c.get("win_rate", 0) * 100, 1)
                sell_tr = c.get("trades", 0)
            except Exception:
                pass

        rows.append({
            "pair": pair,
            "buy_wr": buy_wr,
            "buy_tr": buy_tr,
            "sell_wr": sell_wr,
            "sell_tr": sell_tr,
        })
    return rows


def get_heartbeat():
    """Parse the training log for the latest activity."""
    if not LOG_FILE.exists():
        return "No training log found."
    lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    starting = ""
    attempt = ""
    for line in reversed(lines):
        if not attempt and "Attempt" in line and "/" in line:
            attempt = line.strip().split(" - ")[-1]
        if not starting and "Starting Specialist Factory" in line:
            starting = line.strip().split(" - ")[-1]
        if starting and attempt:
            break
    return f"{starting} | {attempt}" if starting else "Idle"


def generate_markdown(rows, heartbeat):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Compute summary stats
    full = sum(1 for r in rows if r["buy_wr"] != "—" and r["sell_wr"] != "—")
    partial = sum(1 for r in rows if (r["buy_wr"] != "—") != (r["sell_wr"] != "—"))
    total_models = sum(1 for r in rows if r["buy_wr"] != "—") + sum(1 for r in rows if r["sell_wr"] != "—")

    all_wrs = []
    total_trades = 0
    best_name, best_wr = "", 0
    deepest_name, deepest_tr = "", 0

    for r in rows:
        for side, wr_key, tr_key in [("BUY", "buy_wr", "buy_tr"), ("SELL", "sell_wr", "sell_tr")]:
            if r[wr_key] != "—":
                wr = float(r[wr_key])
                tr = int(r[tr_key])
                all_wrs.append(wr)
                total_trades += tr
                if wr > best_wr:
                    best_wr = wr
                    best_name = f"{r['pair']} {side}"
                if tr > deepest_tr:
                    deepest_tr = tr
                    deepest_name = f"{r['pair']} {side}"

    avg_wr = round(sum(all_wrs) / len(all_wrs), 1) if all_wrs else 0

    # Build table rows
    table_lines = []
    for i, r in enumerate(rows, 1):
        bw = f"{r['buy_wr']}%" if r["buy_wr"] != "—" else "—"
        bt = f"{r['buy_tr']:,}" if r["buy_tr"] != "—" else "—"
        sw = f"{r['sell_wr']}%" if r["sell_wr"] != "—" else "—"
        st2 = f"{r['sell_tr']:,}" if r["sell_tr"] != "—" else "—"

        if r["buy_wr"] != "—" and r["sell_wr"] != "—":
            status = "✅ Full"
        elif r["buy_wr"] != "—":
            status = "⚠️ BUY only"
        elif r["sell_wr"] != "—":
            status = "⚠️ SELL only"
        else:
            status = "❌ Pending"

        table_lines.append(f"| {i} | **{r['pair']}** | {bw} | {bt} | {sw} | {st2} | {status} |")

    table = "\n".join(table_lines)

    # Partial pairs list
    partial_names = [r["pair"] for r in rows if (r["buy_wr"] != "—") != (r["sell_wr"] != "—")]
    partial_str = ", ".join(partial_names) if partial_names else "None"

    md = f"""# 📊 Golden Signal Fleet — Full Metrics
**Last Update**: {now} | **Target**: 90% Win Rate | **Search**: 20 Attempts/Model

## ⚡ Training Heartbeat
`{heartbeat}`

---

## 🏆 Full Fleet Metrics

| # | Pair | BUY Win% | BUY Trades | SELL Win% | SELL Trades | Status |
|:--|:---|:---:|:---:|:---:|:---:|:---|
{table}

---

## 📊 Summary
| Metric | Value |
|:---|:---|
| **Total Pairs** | {len(rows)} |
| **Fully Certified (BUY+SELL)** | {full} / {len(rows)} |
| **Partial (one direction)** | {partial} ({partial_str}) |
| **Total Specialist Models** | {total_models} |
| **Fleet-Wide Avg Win Rate** | {avg_wr}% |
| **Total Verified Trades** | {total_trades:,} |
| **Best Model** | {best_name} — {best_wr}% |
| **Deepest Model** | {deepest_name} — {deepest_tr:,} trades |

---

## 🔄 Refresh this report
```powershell
venv\\scripts\\python.exe scripts\\generate_training_report.py
```
"""
    return md


if __name__ == "__main__":
    rows = scan_models()
    hb = get_heartbeat()
    md = generate_markdown(rows, hb)

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_PATH, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"✅ Report updated: {ARTIFACT_PATH}")
    print(f"   Models: {sum(1 for r in rows if r['buy_wr'] != '—') + sum(1 for r in rows if r['sell_wr'] != '—')}")
    print(f"   Heartbeat: {hb}")
