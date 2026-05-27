import os
import sys
import hydra
import lightning.pytorch as pl
import numpy as np
import torch
from pathlib import Path
sys.path.append(os.getcwd())
from ocr.lightning_modules import get_pl_modules_by_cfg  # noqa: E402

CONFIG_DIR = os.environ.get('OP_CONFIG_DIR') or '../configs'


class ProbMapSaverModule(pl.LightningModule):
    def __init__(self, model_module, prob_maps_dir):
        super().__init__()
        self.model = model_module.model
        self.prob_maps_dir = Path(prob_maps_dir)
        self.prob_maps_dir.mkdir(parents=True, exist_ok=True)
        self.saved_count = 0

    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            pred = self.model(return_loss=False, **batch)

        prob_maps = pred['prob_maps'].cpu().numpy()  # (N, 1, H, W)
        inverse_matrices = batch['inverse_matrix']   # list of ndarray (N,)
        filenames = batch['image_filename']
        images = batch['images'].cpu().numpy()       # (N, C, H, W) - shape 복원용

        for i, filename in enumerate(filenames):
            stem = Path(filename).stem
            save_data = {
                'prob_map': prob_maps[i],            # (1, H, W)
                'inverse_matrix': inverse_matrices[i],
                'image_shape': images[i].shape,      # (C, H, W)
                'filename': filename,
            }
            np.save(self.prob_maps_dir / f"{stem}.npy", save_data)
            self.saved_count += 1

        return pred

    def on_predict_epoch_end(self):
        print(f"\n저장 완료: {self.prob_maps_dir}")
        print(f"총 저장된 파일 수: {self.saved_count}")


@hydra.main(config_path=CONFIG_DIR, config_name='predict', version_base='1.2')
def save_prob_maps(config):
    pl.seed_everything(config.get("seed", 42), workers=True)

    model_module, data_module = get_pl_modules_by_cfg(config)

    prob_maps_dir = config.get("prob_maps_dir", "outputs/prob_maps/model")
    saver_module = ProbMapSaverModule(model_module, prob_maps_dir)

    trainer = pl.Trainer(logger=False)

    ckpt_path = config.get("checkpoint_path")
    assert ckpt_path, "checkpoint_path must be provided"

    trainer.predict(
        saver_module,
        data_module,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    save_prob_maps()
