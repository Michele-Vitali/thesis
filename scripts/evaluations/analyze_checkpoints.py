import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_ROOT = ROOT / "adapters/openvla"      # dove stanno i *_log.json e (di default) gli adapter LoRA
DEFAULT_QUEUE = ADAPTER_ROOT / "evaluation_queue.json"

# Nome cartella adapter: "<exp>--<step>_chkpt..."  (exp = nome run)
CHKPT_RE = re.compile(r"^(?P<exp>.+)--(?P<step>\d+)_chkpt")
# Fallback: "<exp>/checkpoint-<step>" oppure "<exp>/step_<step>" (exp = cartella padre)
ALT_RE = re.compile(r"^(?:checkpoint|step)[-_]?(?P<step>\d+)")
VERDICT_ORDER = {"SICURO": 0, "POTREI": 1, "NON DOVREI": 2, "INCOMPLETO": 3}

# Versione dataset nel run_name: "v4", "v4.0", "v3_1"... (non deve essere parte di una parola più lunga)
VERSION_RE = re.compile(r"(?<![A-Za-z0-9])v(?P<num>\d+(?:[._]\d+)*)(?![A-Za-z0-9])", re.IGNORECASE)
NO_VERSION = "senza-versione"


def version_of(run_name: str) -> str:
    """Ritorna la versione dataset (es. 'v4', 'v3.1') estratta dal run_name, o NO_VERSION."""
    m = VERSION_RE.search(run_name)
    return "v" + m["num"].replace("_", ".") if m else NO_VERSION


def version_key(version: str):
    """Ordina le versioni dalla più recente; 'senza-versione' per ultima."""
    if version == NO_VERSION:
        return (1, ())
    parts = [int(x) for x in version[1:].split(".")]
    parts += [0] * (4 - len(parts))          # v4 == v4.0 per l'ordinamento
    return (0, tuple(-x for x in parts))

# ---------------------------------------------------------------- POLICY
# Solo L1 e accuracy decidono il verdetto.
# (soglia, direzione): "lower" -> passa se valore < soglia ; "higher" -> passa se valore > soglia
POLICY = {
    "l1":  (0.03, "lower"),
    "acc": (0.50, "higher"),
}

# Metriche solo informative (mostrate in tabella e salvate nella coda, ma non influenzano il verdetto).
# Quando avrai dei target, basta spostarle in POLICY.
INFO_METRICS = ("xyz", "rpy")

# Regola di aggregazione:
#   SICURO     -> tutte le metriche della policy passano
#   POTREI     -> almeno una passa (ma non tutte)
#   NON DOVREI -> nessuna passa

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


def rate(metric: str, value):
    """Ritorna 'ok' | 'ko' | 'n/d' per le metriche della policy."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/d"
    threshold, direction = POLICY[metric]
    passed = value < threshold if direction == "lower" else value > threshold
    return "ok" if passed else "ko"


def verdict(ratings: dict) -> str:
    vals = list(ratings.values())
    if "n/d" in vals:
        return "INCOMPLETO"
    n_ok = vals.count("ok")
    if n_ok == len(vals):
        return "SICURO"
    if n_ok >= 1:
        return "POTREI"
    return "NON DOVREI"


def index_adapters(adapters_dir: Path):
    """Lista (path, exp, step) di tutte le cartelle adapter LoRA (contengono adapter_config.json)."""
    out = []
    if not adapters_dir.is_dir():
        return out
    for cfg in adapters_dir.rglob("adapter_config.json"):
        p = cfg.parent
        m = CHKPT_RE.match(p.name)
        if m:
            out.append((p, m["exp"], int(m["step"])))
            continue
        m = ALT_RE.match(p.name)
        if m:
            out.append((p, p.parent.name, int(m["step"])))
    return out


def base_model_of(adapter_path: Path):
    """Legge il modello base da adapter_config.json (serve per il merge a runtime)."""
    try:
        with open(adapter_path / "adapter_config.json") as f:
            return json.load(f).get("base_model_name_or_path")
    except (OSError, json.JSONDecodeError):
        return None


def match_checkpoint(run_name: str, ckpt_step, index):
    """Cerca la cartella dell'adapter per (run, step). Ritorna Path o None."""
    try:
        ckpt_step = int(ckpt_step)
    except (TypeError, ValueError):
        return None
    cands = []
    for path, name, step in index:
        if step != ckpt_step:
            continue
        if name == run_name:
            rank = 0
        elif name.startswith(run_name) or run_name.startswith(name):
            rank = 1
        elif run_name in name or name in run_name:
            rank = 2
        else:
            continue
        cands.append((rank, -path.stat().st_mtime, path))
    if not cands:
        return None
    cands.sort()
    if len([c for c in cands if c[0] == cands[0][0]]) > 1:
        print(f"  [!] Più adapter per {run_name} step {ckpt_step}: uso {cands[0][2]}")
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
    print(f"\n{'=' * 78}\nRUN: {run_name}  [dataset {version_of(run_name)}]  ({len(records)} validation)\n{'=' * 78}")
    if not records:
        print("  Nessun record con metriche val/...")
        return []

    keys = {m: (k if k in records[0] else None) for m, k in VAL_KEYS.items()}
    missing = [m for m, k in keys.items() if k is None]
    if missing:
        print(f"  [!] Chiavi non trovate per: {missing}. Controlla VAL_KEYS (usa --inspect).")
    print("  Chiavi usate: " + ", ".join(f"{m}={k}" for m, k in keys.items()))

    entries = {}
    for r in records:
        vals = {m: (r.get(k) if k else None) for m, k in keys.items()}
        ratings = {m: rate(m, vals[m]) for m in POLICY}   # solo metriche della policy
        step = step_of(r)
        ckpt = match_checkpoint(run_name, step, ckpt_index)
        entries[step] = {
            "run": run_name,
            "version": version_of(run_name),
            "step": step,
            "name": ckpt.name if ckpt else None,
            "adapter_path": str(ckpt.resolve()) if ckpt else None,
            "base_model": base_model_of(ckpt) if ckpt else None,
            "merge_at_runtime": True,
            "on_disk": ckpt is not None,
            "verdict": verdict(ratings),
            "metrics": vals,
            "ratings": ratings,
        }
    rows = sorted(entries.values(), key=sort_key)

    def fmt(e, m):
        v = e["metrics"][m]
        if v is None:
            return "    n/d   "
        mark = {"ok": "+", "ko": "-"}.get(e["ratings"].get(m), " ")   # xyz/rpy: nessun segno
        return f"{v:8.4f}{mark} "

    print(f"\n  {'step':>8} | {'L1':>9} | {'ACC':>9} | {'XYZ':>9} | {'RPY':>9} | {'verdetto':<10} | adapter")
    print("  " + "-" * 78)
    for e in rows:
        cells = " | ".join(fmt(e, m) for m in ("l1", "acc", "xyz", "rpy"))
        print(f"  {str(e['step']):>8} | {cells} | {e['verdict']:<10} | {'ok' if e['on_disk'] else 'NON TROVATO'}")
    print("  (+ = soglia rispettata, - = non rispettata; XYZ/RPY solo informative)")

    for label in ("SICURO", "POTREI", "NON DOVREI"):
        steps = [str(e["step"]) for e in rows if e["verdict"] == label]
        print(f"  {label:<11}: {', '.join(steps) if steps else '-'}")
    return rows


def sort_key(e: dict):
    """Priorità: versione dataset (più recente prima), verdetto, L1 più bassa, accuracy più alta."""
    l1 = e["metrics"].get("l1")
    acc = e["metrics"].get("acc")
    return (
        version_key(e["version"]),
        VERDICT_ORDER[e["verdict"]],
        l1 if l1 is not None else float("inf"),
        -acc if acc is not None else float("inf"),
    )


def save_queue(entries: list, out: Path, adapters_dir: Path):
    entries = sorted(entries, key=sort_key)
    versions = sorted({e["version"] for e in entries}, key=version_key)
    queue = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "adapters_dir": str(adapters_dir),
        "policy": {
            "thresholds": {m: {"soglia": t, "direzione": d} for m, (t, d) in POLICY.items()},
            "info_metrics": list(INFO_METRICS),
        },
        "counts": {v: sum(e["verdict"] == v for e in entries) for v in VERDICT_ORDER},
        "counts_by_version": {
            ver: {v: sum(e["version"] == ver and e["verdict"] == v for e in entries) for v in VERDICT_ORDER}
            for ver in versions
        },
        # versione -> verdetto -> lista di checkpoint (già ordinati per priorità di valutazione)
        "by_version": {
            ver: {v: [e for e in entries if e["version"] == ver and e["verdict"] == v] for v in VERDICT_ORDER}
            for ver in versions
        },
        "models": entries,   # lista piatta: versione più recente prima, poi verdetto
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=4, ensure_ascii=False)

    print(f"\n{'=' * 78}\nCODA DI VALUTAZIONE salvata in: {out}")
    for ver in versions:
        print(f"\n  Dataset {ver}")
        for v in ("SICURO", "POTREI", "NON DOVREI", "INCOMPLETO"):
            group = [e for e in entries if e["version"] == ver and e["verdict"] == v]
            no_disk = sum(not e["on_disk"] for e in group)
            extra = f"  ({no_disk} senza adapter su disco)" if no_disk else ""
            print(f"    {v:<11}: {len(group)}{extra}")


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
    ap.add_argument("--adapters-dir", "--runs-dir", dest="adapters_dir", type=Path, default=ADAPTER_ROOT,
                    help="Cartella (cercata ricorsivamente) con gli adapter LoRA (contengono adapter_config.json)")
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

    if not args.adapters_dir.is_dir():
        print(f"[!] Cartella adapter non trovata: {args.adapters_dir} (usa --adapters-dir)")
    ckpt_index = index_adapters(args.adapters_dir)
    print(f"Trovati {len(ckpt_index)} adapter LoRA in {args.adapters_dir}")

    all_entries = []
    for f in files:
        all_entries.extend(analyze_file(f, ckpt_index))
    save_queue(all_entries, args.out, args.adapters_dir)


if __name__ == "__main__":
    main()