import argparse
import json
import os
import sys
import numpy as np
import torch
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
sys.path.append(os.getcwd())
from tqdm import tqdm
from ocr.models.head.db_postprocess import DBPostProcessor  # noqa: E402


def load_prob_maps(prob_maps_dir):
    # 저장된 npy 파일들을 filename 기준으로 로드
    prob_maps_dir = Path(prob_maps_dir)
    data = {}
    for npy_file in sorted(prob_maps_dir.glob("*.npy")):
        saved = np.load(npy_file, allow_pickle=True).item()
        filename = saved['filename']
        data[filename] = saved
    print(f"로드 완료: {prob_maps_dir} ({len(data)}개)")
    return data


def ensemble_and_postprocess(dir_a, dir_b, weight_a, weight_b, output_path,
    # Note by Karyx💫: This part of the code is omitted to protect my intellectual property. Please implement your own logic here.
    ):

    print(f"\n[1/3] prob_maps 로드 중...")
    data_a = load_prob_maps(dir_a)
    data_b = load_prob_maps(dir_b)

    filenames_a = set(data_a.keys())
    filenames_b = set(data_b.keys())
    filenames = filenames_a & filenames_b  # 교집합만 앙상블
    only_a = filenames_a - filenames_b
    only_b = filenames_b - filenames_a

    if only_a:
        print(f"⚠ dir_a에만 있는 파일 {len(only_a)}개 → dir_a 단독 사용")
    if only_b:
        print(f"⚠ dir_b에만 있는 파일 {len(only_b)}개 → dir_b 단독 사용")
    print(f"앙상블 대상: {len(filenames)}개")

    print(f"\n[2/3] Weighted averaging + postprocess 중... (weight_a={weight_a}, weight_b={weight_b})")

    postprocessor = DBPostProcessor(
        thresh=thresh,
        box_thresh=box_thresh,
        max_candidates=max_candidates,
        use_polygon=use_polygon,
        polygon_unclip_ratio=polygon_unclip_ratio,
        box_unclip_ratio=box_unclip_ratio,
    )

    submission = OrderedDict(images=OrderedDict())
    total_boxes = 0

    all_filenames = sorted(filenames | only_a | only_b)

    for filename in tqdm(all_filenames):
        if filename in filenames:
            # weighted average
            pm_a = torch.tensor(data_a[filename]['prob_map'])  # (1, H, W)
            pm_b = torch.tensor(data_b[filename]['prob_map'])  # (1, H, W)
            prob_map = weight_a * pm_a + weight_b * pm_b
            inverse_matrix = data_a[filename]['inverse_matrix']
            image_shape = data_a[filename]['image_shape']
        elif filename in filenames_a:
            prob_map = torch.tensor(data_a[filename]['prob_map'])
            inverse_matrix = data_a[filename]['inverse_matrix']
            image_shape = data_a[filename]['image_shape']
        else:
            prob_map = torch.tensor(data_b[filename]['prob_map'])
            inverse_matrix = data_b[filename]['inverse_matrix']
            image_shape = data_b[filename]['image_shape']

        # postprocessor.represent()에 맞게 batch/pred 구성
        # images shape: (1, C, H, W) dummy (size 정보만 필요)
        dummy_images = torch.zeros(1, *image_shape)
        batch = {
            'images': dummy_images,
            'inverse_matrix': [inverse_matrix],
        }
        pred = {'prob_maps': prob_map.unsqueeze(0)}  # (1, 1, H, W)

        boxes, _ = postprocessor.represent(batch, pred)
        boxes = boxes[0]  # batch size 1
        total_boxes += len(boxes)

        words = OrderedDict()
        for idx, box in enumerate(boxes):
            words[f'{idx + 1:04}'] = OrderedDict(points=box)
        submission['images'][filename] = OrderedDict(words=words)

    print(f"\n[3/3] JSON 저장 중...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=4)

    print(f"저장 완료: {output_path}")
    print(f"총 이미지: {len(submission['images'])}")
    print(f"총 박스 수: {total_boxes}")
    print(f"평균 박스 수: {total_boxes / len(submission['images']):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_a', type=str, required=True,
                        help='모델 A prob_maps 저장 디렉토리 (HRNet-W44)')
    parser.add_argument('--dir_b', type=str, required=True,
                        help='모델 B prob_maps 저장 디렉토리 (ConvNeXt-Base)')
    parser.add_argument('--weight_a', type=float, default=0.6,
                        help='모델 A 가중치 (default: 0.6)')
    parser.add_argument('--weight_b', type=float, default=0.4,
                        help='모델 B 가중치 (default: 0.4)')
    parser.add_argument('--output', type=str,
                        default=f"outputs/ocr_training/submissions/ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        help='앙상블 결과 JSON 저장 경로')
    # postprocess 파라미터 (db_head.yaml과 동일하게)
    parser.add_argument('--thresh', type=float, default=0.15)
    parser.add_argument('--box_thresh', type=float, default=0.40)
    parser.add_argument('--max_candidates', type=int, default=1000)
    parser.add_argument('--polygon_unclip_ratio', type=float, default=1.31)
    parser.add_argument('--box_unclip_ratio', type=float, default=1.87)
    args = parser.parse_args()

    ensemble_and_postprocess(
        dir_a=args.dir_a,
        dir_b=args.dir_b,
        weight_a=args.weight_a,
        weight_b=args.weight_b,
        output_path=args.output,
        thresh=args.thresh,
        box_thresh=args.box_thresh,
        max_candidates=args.max_candidates,
        polygon_unclip_ratio=args.polygon_unclip_ratio,
        box_unclip_ratio=args.box_unclip_ratio,
    )
