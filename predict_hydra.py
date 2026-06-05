from __future__ import annotations

import argparse
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from predict import main as predict_main


def _none_if_missing(value: Any) -> Any:
    return None if value in {"", "null", "None"} else value


def build_predict_args(cfg: DictConfig) -> argparse.Namespace:
    data = OmegaConf.to_container(cfg, resolve=True)
    paths = data["paths"]
    model = data["model"]
    device = data["device"]
    return argparse.Namespace(
        image_dir=paths["image_dir"],
        output=paths["output"],
        checkpoint=paths["checkpoint"],
        classes=paths["classes"],
        score_threshold=float(model["score_threshold"]),
        nms_threshold=float(model["nms_threshold"]),
        backbone=model["backbone"],
        custom=bool(model["custom"]),
        min_size=int(model["min_size"]),
        max_size=int(model["max_size"]),
        anchor_sizes=model.get("anchor_sizes", "") or "",
        anchor_ratios=model.get("anchor_ratios", "") or "",
        device=_none_if_missing(device.get("device")),
    )


@hydra.main(version_base=None, config_path="configs", config_name="predict")
def main(cfg: DictConfig) -> None:
    predict_main(build_predict_args(cfg))


if __name__ == "__main__":
    main()
