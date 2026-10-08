"""Export matching fitting cases as CSV; compare geometry before drawing timing conclusions."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before, after = [json.loads(p.read_text()) for p in (args.before, args.after)]
    if before['environment'].get('schema_version', 1) != after['environment'].get('schema_version', 1):
        raise ValueError('Curve schema differs; rerun both benchmarks with the same script version')
    for key in ("targets_sha256", "steps", "repeats", "source", "torch", "device"):
        if before["environment"][key] != after["environment"][key]:
            raise ValueError(f"Incomparable benchmark metadata: {key}")

    def key(row):
        return tuple(row[k] for k in ("side", "batch", "task", "mode"))

    old, new = [{key(row): row for row in result["cases"]} for result in (before, after)]
    if not old or len(old) != len(before['cases']) or len(new) != len(after['cases']):
        raise ValueError('Case lists must be nonempty and contain no duplicate keys')
    if old.keys() != new.keys():
        raise ValueError("Case lists differ")
    rows = []
    for k, a in old.items():
        b = new[k]
        if a['threshold_mm'] != b['threshold_mm']:
            raise ValueError(f'Error thresholds differ: {k}')
        rows.append({"side": k[0], "batch": k[1], "task": k[2], "mode": k[3],
                     "before_step_ms": a["fit_step_ms"], "after_step_ms": b["fit_step_ms"],
                     "before_final_rmse_mm": a["final_rmse_mm"], "after_final_rmse_mm": b["final_rmse_mm"],
                     "error_difference_mm": abs(a["final_rmse_mm"] - b["final_rmse_mm"]),
                     "before_steps_to_threshold": a["steps_to_threshold"], "after_steps_to_threshold": b["steps_to_threshold"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"cases": len(rows), "max_final_error_difference_mm": max(r["error_difference_mm"] for r in rows),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
