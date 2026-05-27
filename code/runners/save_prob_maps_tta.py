import os
import sys
import hydra
import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
sys.path.append(os.getcwd())
from ocr.lightning_modules import get_pl_modules_by_cfg  # noqa: E402

CONFIG_DIR = os.environ.get('OP_CONFIG_DIR') or '../configs'


class ProbMapTTASaverModule(pl.LightningModule):
    def __init__(self, model_module, prob_maps_dir):
        super().__init__()
        self.model = model_module.model
        self.prob_maps_dir = Path(prob_maps_dir)
        self.prob_maps_dir.mkdir(parents=True, exist_ok=True)
        self.saved_count = 0

    def predict_step(self, batch, batch_idx):
        images = batch['images']  # (N, C, H, W)
        _, _, H, W = images.shape

        with torch.no_grad():
            # 1. 원본 추론
            pred_orig = self.model(return_loss=False, **batch)
            prob_orig = pred_orig['prob_maps']  # (N, 1, H, W)

            # 2. hflip 추론
            images_flipped = torch.flip(images, dims=[3])
            batch_flipped = {**batch, 'images': images_flipped}
            pred_flipped = self.model(return_loss=False, **batch_flipped)
            prob_flipped_restored = torch.flip(pred_flipped['prob_maps'], dims=[3])

            # 3. multi-scale 추론 (scale=1600)
            scale_factor = 1600 / max(H, W)
            new_H = int(H * scale_factor)
            new_W = int(W * scale_factor)
            images_large = F.interpolate(images, size=(new_H, new_W),
                                         mode='bilinear', align_corners=False)
            pad_h = 1600 - new_H
            pad_w = 1600 - new_W
            images_large = F.pad(images_large, (0, pad_w, 0, pad_h))
            pred_large = self.model(return_loss=False, **{**batch, 'images': images_large})
            prob_large = F.interpolate(pred_large['prob_maps'][:, :, :new_H, :new_W],
                                       size=(H, W), mode='bilinear', align_corners=False)

            prob_avg = ...
            # Note by Karyx💫: This part of the code is omitted to protect my intellectual property. Please implement your own logic here.

        prob_maps = prob_avg.cpu().numpy()
        inverse_matrices = batch['inverse_matrix']
        filenames = batch['image_filename']
        images_np = images.cpu().numpy()

        for i, filename in enumerate(filenames):
            stem = Path(filename).stem
            save_data = {
                'prob_map': prob_maps[i],
                'inverse_matrix': inverse_matrices[i],
                'image_shape': images_np[i].shape,
                'filename': filename,
            }
            np.save(self.prob_maps_dir / f"{stem}.npy", save_data)
            self.saved_count += 1

        return pred_orig

    def on_predict_epoch_end(self):
        print(f"\n저장 완료 (TTA hflip+scale1600 2:2:1): {self.prob_maps_dir}")
        print(f"총 저장된 파일 수: {self.saved_count}")


@hydra.main(config_path=CONFIG_DIR, config_name='predict', version_base='1.2')
def save_prob_maps_tta(config):
    pl.seed_everything(config.get("seed", 42), workers=True)

    model_module, data_module = get_pl_modules_by_cfg(config)

    prob_maps_dir = config.get("prob_maps_dir", "outputs/prob_maps/model_tta")
    saver_module = ProbMapTTASaverModule(model_module, prob_maps_dir)

    trainer = pl.Trainer(logger=False)

    ckpt_path = config.get("checkpoint_path")
    assert ckpt_path, "checkpoint_path must be provided"

    trainer.predict(
        saver_module,
        data_module,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    save_prob_maps_tta()
