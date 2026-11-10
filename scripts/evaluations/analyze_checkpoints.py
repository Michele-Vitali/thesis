"""
Analizza i log JSON esportati da wandb e classifica i checkpoint
(record che contengono le metriche "val/...") in base a una policy a soglie.

Genera anche una "coda" JSON (evaluation_queue.json) con i checkpoint divisi per
categoria e il path dell'adapter, da dare in pasto a evaluate_model.py (--queue).

Uso:
    python analyze_checkpoints.py                 # cerca in ADAPTER_ROOT/*/*_log.json
    python analyze_checkpoints.py path/a/log.json # file singolo
    python analyze_checkpoints.py --inspect       # stampa le chiavi di un record validation
    python analyze_checkpoints.py --out coda.json --runs-dir /path/runs/openvla
"""
import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_ROOT = ROOT / "adapters/openvla"      # dove stanno i *_log.json
RUNS_DIR = ROOT / "runs/openvla"              # dove stanno le cartelle dei checkpoint (<exp>--<step>_chkpt)
DEFAULT_QUEUE = ADAPTER_ROOT / "evaluation_queue.json"
CHKPT_RE = re.compile(r"^(?P<exp>.+)--(?P<step>\d+)_chkpt")
VERDICT_ORDER = {"SICURO": 0, "POTREI": 1, "NON DOVREI": 2, "INCOMPLETO": 3}

# ---------------------------------------------------------------- POLICY
# (soglia_ottima, soglia_pessima, direzione)
#   "lower": valore < ottima -> ottima ; valore > pessima -> pessima ; in mezzo buona
#   "higher": valore > ottima -> ottima ; valore < pessima -> pessima ; in mezzo buona
POLICY = {
    "l1":  (0.03, 0.045, "lower"),
    "xyz": (3.50, 5.0,   "lower"),
    "rpy": (1.60, 2.50,  "lower"),
    "acc": (0.5,  0.4,   "higher"),
}

# Regola di aggregazione
SAFE_MIN_OTTIME = 2        # SICURO: 0 pessime e almeno N ottime
MAYBE_MAX_PESSIME = 1      # POTREI: al massimo N pessime
# oltre -> NON DOVREI

# Chiavi wandb delle metriche di validation
VAL_KEYS = {
    "l1":  "val/l1_loss",
    "xyz": "val/loss_xyz",
    "rpy": "val/loss_rpy",
    "acc": "val/action_accuracy",
}

# Chiavi candidate per identificare il checkpoint
STEP_KEYS = ["step", "_step", "global_step", "epoch"]
# ----------------------------------------------------------------------


def find_key(record: dict, metric: str):
    """Ritorna la chiave val/... della metrica se presente nel record."""
    key = VAL_KEYS[metric]
    return key if key in record else None


def rate(metric: str, value):
    """Ritorna 'ottima' | 'buona' | 'pessima' | 'n/d'."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/d"
    good, bad, direction = POLICY[metric]
    if direction == "lower":
        if value < good:
            return "ottima"
        return "buona" if value <= bad else "pessima"
    if value > good:
        return "ottima"
    return "buona" if value >= bad else "pessima"


def verdict(ratings: dict) -> str:
    vals = list(ratings.values())
    if "n/d" in vals:
        return "INCOMPLETO"
    n_ok = vals.count("ottima")
    n_bad = vals.count("pessima")
    if n_bad == 0 and n_ok >= SAFE_MIN_OTTIME:
        return "SICURO"
    if n_bad <= MAYBE_MAX_PESSIME:
        return "POTREI"
    return "NON DOVREI"


def index_checkpoints(runs_dir: Path):
    """Lista (path, exp, step) di tutte le cartelle *_chkpt presenti su disco."""
    out = []
    if runs_dir.is_dir():
        for p in runs_dir.iterdir():
            m = CHKPT_RE.match(p.name)
            if p.is_dir() and m:
                out.append((p, m["exp"], int(m["step"])))
    return out


def match_checkpoint(run_name: str, step, index):
    """Cerca la cartella del checkpoint per (run, step). Ritorna Path o None."""
    try:
        step = int(step)
    except (TypeError, ValueError):
        return None
    cands = []
    for p, exp, st in index:
        if st != step:
            continue
        if exp == run_name:
            rank = 0
        elif exp.startswith(run_name) or run_name.startswith(exp):
            rank = 1
        elif run_name in exp or exp in run_name:
            rank = 2
        else:
            continue
        cands.append((rank, -p.stat().st_mtime, p))
    if not cands:
        return None
    cands.sort()
    if len([c for c in cands if c[0] == cands[0][0]]) > 1:
        print(f"  [!] Più checkpoint per {run_name} step {step}: uso {cands[0][2].name}")
    return cands[0][2]


def load_records(path: Path):
    with open(path) as f:
        data = json.load(f)
    # wandb scrive un dict completo ad ogni step: è "validation" solo se contiene i val/...
    return [r for r in data
            if isinstance(r, dict) and any(r.get(k) is not None for k in VAL_KEYS.values())]


def step_of(record: dict):
    for k in STEP_KEYS:
        if k in record and record[k] is not None:
            return record[k]
    return "?"


def analyze_file(path: Path, ckpt_index):
    run_name = path.stem.removesuffix("_log")
    records = load_records(path)
    print(f"\n{'=' * 78}\nRUN: {run_name}  ({len(records)} validation)\n{'=' * 78}")
    if not records:
        print("  Nessun record con metriche val/...")
        return []

    keys = {m: find_key(records[0], m) for m in POLICY}
    missing = [m for m, k in keys.items() if k is None]
    if missing:
        print(f"  [!] Chiavi non trovate per: {missing}. Controlla VAL_KEYS (usa --inspect).")
    print("  Chiavi usate: " + ", ".join(f"{m}={k}" for m, k in keys.items()))

    entries = {}
    for r in records:
        vals = {m: (r.get(k) if k else None) for m, k in keys.items()}
        ratings = {m: rate(m, v) for m, v in vals.items()}
        step = step_of(r)
        ckpt = match_checkpoint(run_name, step, ckpt_index)
        entries[step] = {                      # stesso step ripetuto -> vale l'ultimo record
            "run": run_name,
            "step": step,
            "name": ckpt.name if ckpt else None,
            "adapter_path": str(ckpt.resolve()) if ckpt else None,
            "on_disk": ckpt is not None,
            "verdict": verdict(ratings),
            "metrics": vals,
            "ratings": ratings,
        }
    rows = sorted(entries.values(), key=sort_key)

    def fmt(m, v, rt):
        return "   n/d   " if v is None else f"{v:8.4f}{rt[0].upper()}"

    print(f"\n  {'step':>8} | {'L1':>9} | {'XYZ':>9} | {'RPY':>9} | {'ACC':>9} | {'verdetto':<10} | ckpt")
    print("  " + "-" * 78)
    for e in rows:
        cells = " | ".join(fmt(m, e["metrics"][m], e["ratings"][m]) for m in ("l1", "xyz", "rpy", "acc"))
        print(f"  {str(e['step']):>8} | {cells} | {e['verdict']:<10} | {'ok' if e['on_disk'] else 'NON TROVATO'}")
    print("  (O=ottima, B=buona, P=pessima)")

    for label in ("SICURO", "POTREI", "NON DOVREI"):
        steps = [str(e["step"]) for e in rows if e["verdict"] == label]
        print(f"  {label:<11}: {', '.join(steps) if steps else '-'}")
    return rows


def sort_key(e: dict):
    """Priorità: verdetto, più 'ottime', L1 più bassa."""
    n_ok = sum(v == "ottima" for v in e["ratings"].values())
    l1 = e["metrics"].get("l1")
    return (VERDICT_ORDER[e["verdict"]], -n_ok, l1 if l1 is not None else float("inf"))


def save_queue(entries: list, out: Path, runs_dir: Path):
    entries = sorted(entries, key=sort_key)
    queue = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runs_dir": str(runs_dir),
        "policy": {
            "thresholds": {m: {"ottima": g, "pessima": b, "direzione": d} for m, (g, b, d) in POLICY.items()},
            "SAFE_MIN_OTTIME": SAFE_MIN_OTTIME,
            "MAYBE_MAX_PESSIME": MAYBE_MAX_PESSIME,
        },
        "counts": {v: sum(e["verdict"] == v for e in entries) for v in VERDICT_ORDER},
        "models": entries,   # già ordinati per priorità di valutazione
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=4, ensure_ascii=False)

    print(f"\n{'=' * 78}\nCODA DI VALUTAZIONE salvata in: {out}")
    for v in ("SICURO", "POTREI", "NON DOVREI", "INCOMPLETO"):
        group = [e for e in entries if e["verdict"] == v]
        no_disk = sum(not e["on_disk"] for e in group)
        extra = f"  ({no_disk} senza checkpoint su disco)" if no_disk else ""
        print(f"  {v:<11}: {len(group)}{extra}")


def inspect(path: Path):
    records = load_records(path)
    if not records:
        print(f"{path}: nessun record con val/...")
        return
    print(f"{path}: chiavi del primo record con val/...")
    for k, v in records[0].items():
        print(f"  {k!r}: {v!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="JSON di log (default: tutti in ADAPTER_ROOT)")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help="Cartella con i checkpoint *_chkpt")
    ap.add_argument("--out", type=Path, default=DEFAULT_QUEUE, help="File JSON di output con la coda")
    args = ap.parse_args()

    files = [Path(f) for f in args.files] or sorted(ADAPTER_ROOT.glob("*/*_log.json"))
    if not files:
        print(f"Nessun file trovato in {ADAPTER_ROOT}")
        return

    if args.inspect:
        for f in files:
            inspect(f)
        return

    if not args.runs_dir.is_dir():
        print(f"[!] Cartella checkpoint non trovata: {args.runs_dir} (usa --runs-dir)")
    ckpt_index = index_checkpoints(args.runs_dir)

    all_entries = []
    for f in files:
        all_entries.extend(analyze_file(f, ckpt_index))
    save_queue(all_entries, args.out, args.runs_dir)


if __name__ == "__main__":
    main()