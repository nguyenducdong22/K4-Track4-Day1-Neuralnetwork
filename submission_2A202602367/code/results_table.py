from __future__ import annotations
import json, os, openpyxl
from pathlib import Path

FORMULA_COLS = {"step0_gap_vs_lnC", "gap_val_minus_train", "delta_val_f1_vs_base", "beyond_noise"}

def save_result(result: dict, results_dir: str = "../results") -> str:
    os.makedirs(results_dir, exist_ok=True)
    exp_id = result["cfg"]["exp_id"]
    file_path = os.path.join(results_dir, f"{exp_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump({"cfg": result["cfg"], "history": result["history"], "summary": result["summary"]}, f, indent=2, ensure_ascii=False)
    return file_path

def load_results(results_dir: str = "../results") -> list[dict]:
    p = Path(results_dir)
    if not p.exists(): return []
    res = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(p.glob("*.json"))]
    res.sort(key=lambda x: x.get("cfg", {}).get("exp_id", ""))
    return res

def to_row(result: dict, eval_scores: dict | None = None, notes: str = "") -> dict:
    row = dict(result["cfg"])
    row.update(result.get("summary", {}))
    if isinstance(row.get("hidden"), (list, tuple)):
        row["hidden"] = "-".join(str(h) for h in row["hidden"])
    row["figure_file"] = f"figures/{row['exp_id']}.png"
    row["notes"] = notes
    row["eval_acc"] = eval_scores.get("accuracy", "") if eval_scores else ""
    row["eval_macro_f1"] = eval_scores.get("macro_f1", "") if eval_scores else ""
    return row

def write_xlsx(rows: list[dict], template_path: str, out_path: str) -> None:
    wb = openpyxl.load_workbook(template_path)
    ws = wb["Experiments"]
    col_map = {str(ws.cell(row=1, column=c).value).strip(): c for c in range(1, ws.max_column + 1) if ws.cell(row=1, column=c).value}
    for r_idx, r_data in enumerate(rows, start=2):
        for col_name, c_idx in col_map.items():
            if col_name not in FORMULA_COLS and col_name in r_data:
                ws.cell(row=r_idx, column=c_idx, value=r_data[col_name])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    wb.save(out_path)
    print(f"Đã lưu bảng Excel thành công vào: {out_path}")
