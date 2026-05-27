import json
import os
import cv2
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")
IMAGE_PATH = os.getenv("IMAGE_PATH")
OUTPUT_PATH = os.getenv("OUTPUT_PATH")

def generate_inspection(json_path, image_dir, output_dir=OUTPUT_PATH):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_items = data.get("images", {}).items()
    print(f"총 {len(image_items)}장 시각화 시작...")

    for img_name, img_info in tqdm(image_items):
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        words = img_info.get("words", {})
        box_count = len(words)

        for w_id, w_info in words.items():
            pts = np.array(w_info["points"], dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=1)

        # 파일명에 박스 개수 포함
        name, ext = os.path.splitext(img_name)
        save_name = f"{name}_box_{box_count:03d}{ext}"
        cv2.imwrite(os.path.join(output_dir, save_name), img)

generate_inspection(os.path.join(DATA_PATH, "submission.json"), os.path.join(IMAGE_PATH, "test"))
