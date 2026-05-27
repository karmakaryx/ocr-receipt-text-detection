import json
import os
import cv2
from dotenv import load_dotenv
from pathlib import Path
import sys
sys.path.append(".")
from ocr.utils.ocr_utils import draw_boxes

load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")
IMAGE_PATH = os.getenv("IMAGE_PATH")
OUTPUT_PATH = os.getenv("OUTPUT_PATH")

VAL_IMAGE_DIR = os.path.join(IMAGE_PATH, "val")
VAL_GT_JSON = os.path.join(DATA_PATH, "val.json")
# PRED_JSON = os.path.join(DATA_PATH, "submission_val_tta.json")
OUTPUT_DIR = OUTPUT_PATH

PRED_CONFIGS = {
    "hrnet":    os.path.join(DATA_PATH, "submission_val.json"),
    # "convnext": "outputs/ocr_training/submissions/convnext_val.json",
    "ensemble": os.path.join(DATA_PATH, "submission_val_tta.json"),
}

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

with open(VAL_GT_JSON) as f:
    gt_data = json.load(f)

pred_data = {}
for key, path in PRED_CONFIGS.items():
    if Path(path).exists():
        with open(path) as f:
            pred_data[key] = json.load(f)
    else:
        print(f"[SKIP] {key}: {path} 없음")

# 모델별 missing 리포트 저장용
reports = {key: [] for key in pred_data}

for filename in gt_data["images"]:
    image_path = f"{VAL_IMAGE_DIR}/{filename}"
    if not Path(image_path).exists():
        continue

    gt_words = gt_data["images"][filename].get("words", {})
    gt_polys = [w["points"] for w in gt_words.values() if w.get("points")]
    stem = Path(filename).stem

    for key, data in pred_data.items():
        pred_words = data["images"].get(filename, {}).get("words", {})
        pred_polys = [w["points"] for w in pred_words.values() if w.get("points")]

        diff = len(gt_polys) - len(pred_polys)
        miss_rate = diff / max(len(gt_polys), 1)
        reports[key].append((filename, len(gt_polys), len(pred_polys), diff, miss_rate))

    # 시각화는 ensemble 기준으로 저장 (없으면 hrnet)
    vis_key = "ensemble" if "ensemble" in pred_data else "hrnet"
    pred_polys_vis = [w["points"] for w in pred_data[vis_key]["images"].get(filename, {}).get("words", {}).values()]
    image = draw_boxes(image_path, pred_polys_vis, gt_polys)
    diff_vis = len(gt_polys) - len(pred_polys_vis)
    save_name = f"{stem}_gt{len(gt_polys)}_pred{len(pred_polys_vis)}_diff{diff_vis:+d}.jpg"
    cv2.imwrite(f"{OUTPUT_DIR}/{save_name}", image)

# 모델별 missing 리포트 txt 저장
for key, rows in reports.items():
    rows_sorted = sorted(rows, key=lambda x: x[4], reverse=True)  # miss_rate 내림차순
    report_path = Path(OUTPUT_DIR) / f"missing_{key}.txt"

    with open(report_path, "w") as f:
        f.write(f"{'filename':<55} {'GT':>5} {'Pred':>5} {'Diff':>5} {'Miss%':>7}\n")
        f.write("-" * 80 + "\n")
        for filename, gt, pred, diff, miss in rows_sorted:
            f.write(f"{filename:<55} {gt:>5} {pred:>5} {diff:>5} {miss:>7.1%}\n")
        f.write("-" * 80 + "\n")
        avg_miss = sum(r[4] for r in rows) / max(len(rows), 1)
        total_gt = sum(r[1] for r in rows)
        total_pred = sum(r[2] for r in rows)
        f.write(f"{'전체 평균 miss rate':<55} {total_gt:>5} {total_pred:>5} {total_gt-total_pred:>5} {avg_miss:>7.1%}\n")
    print(f"저장: {report_path}")

print(f"\n시각화 완료: {OUTPUT_DIR}")
