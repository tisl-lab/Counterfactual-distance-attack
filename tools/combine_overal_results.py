import csv
from pathlib import Path
from typing import List, Dict


ALLOWED_METHODS = [
    "NICE",
    "dice_kdtree",
    "dice_gradient",
    "scfe",
]

ALLOWED_ATTACK_METHODS = {f"dist_lrt_local_{m}" for m in ALLOWED_METHODS}


def find_averages_files(root: Path) -> List[Path]:
    attack_results_root = root / "dpnice" / "experiments" / "attack_results"
    return list(attack_results_root.glob("*/averages_results.csv"))


def read_and_filter_rows(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                aset = int(float(str(row.get("attack_set_size", "")).strip()))
            except ValueError:
                continue
            attack_method = str(row.get("attack_method", "")).strip()
            if aset in (500,1000) and attack_method in ALLOWED_ATTACK_METHODS:
                rows.append(row)
    return rows


def write_overal_results(rows: List[Dict[str, str]], header: List[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in header})


def main() -> None:
    
    repo_root = "dpnice/experiments/attack_results"
    out_path = Path(repo_root) / "combined_overal_results.csv"
    all_rows: List[Dict[str, str]] = []
    for dataset in ("acs_income", "heloc", "adult", "compas"):
        in_files = ('dpnice/experiments/attack_results/{}/averages_results.csv' ).format(dataset)
        csv_path = Path(in_files)  # Convert string to Path object

        
        # for csv_path in find_averages_files(repo_root):
            # if csv_path.parts[-3] != dataset:
                # continue
        rows = read_and_filter_rows(csv_path)
        all_rows.extend(rows)
        if all_rows:
            header = list(all_rows[0].keys())
            
            if out_path.exists():
                existing_rows = read_and_filter_rows(out_path)
                all_rows.extend(existing_rows)
        
    write_overal_results(all_rows, header, out_path)

        
if __name__ == "__main__":
    main()
