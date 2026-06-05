1. Install dependencies:

```bash
bash script.sh install
```

2. Download training data:

```bash
bash script.sh download
```

3. Train one named phase:

```bash
WANDB_RUN_NAME=baseline bash script.sh train
USE_WANDB=1 WANDB_RUN_NAME=baseline GPU=0 bash script.sh train
```

The current model uses a custom Faster R-CNN detector with a ResNet-101
backbone. RPN, anchors, proposals, ROI pooling, and detection heads are built in
this repository; only the optional backbone weights come from ImageNet. Use a
new phase name when comparing with older ResNet-50 runs:

Checkpoints created by the previous `torchvision.models.detection` model are
not compatible with this custom implementation; retrain a new run before
predicting.

```bash
USE_WANDB=1 WANDB_RUN_NAME=resnet101-baseline GPU=0 bash script.sh train
```

Backbone and input resize can be changed from `script.sh` environment variables:

```bash
BACKBONE=resnet50 MIN_SIZE=512 MAX_SIZE=768 \
WANDB_RUN_NAME=resnet50-512x768 GPU=0 bash script.sh train

BACKBONE=resnet101 MIN_SIZE=768 MAX_SIZE=1024 \
WANDB_RUN_NAME=resnet101-768x1024 GPU=0 bash script.sh train
```

To focus training on a weak class such as `chair`, enable image-level
oversampling. This keeps the epoch length unchanged, but samples images that
contain `chair` more often:

```bash
OVERSAMPLE_CLASS=chair OVERSAMPLE_FACTOR=2.0 \
BACKBONE=resnet101 MIN_SIZE=512 MAX_SIZE=768 \
AUGMENTATION=1 HORIZONTAL_FLIP_PROBABILITY=0.5 \
COLOR_JITTER_PROBABILITY=0.0 GRAYSCALE_PROBABILITY=0.0 \
WANDB_RUN_NAME=resnet101-512x768-fliponly-chairx2 GPU=0 \
bash script.sh train
```

Use `OVERSAMPLE_FACTOR=1.5` or `2.0` first. Larger values can improve recall
for `chair`, but may increase false positives or hurt stronger classes.

After checking bbox sizes in `notebooks/data_analysis.ipynb`, custom anchor
sizes and ratios can also be overridden:

```bash
ANCHOR_SIZES=64,128,192,256,512 \
ANCHOR_RATIOS=0.33,0.5,1.0,2.0 \
BACKBONE=resnet101 MIN_SIZE=512 MAX_SIZE=768 \
WANDB_RUN_NAME=resnet101-512x768-custom-anchor GPU=0 \
bash script.sh train
```

Training artifacts are stored under:

```text
saved_results/<WANDB_RUN_NAME>/
├── checkpoints/
│   ├── best_model.pth
│   └── last_model.pth
└── logs/
    └── session.log
```

Resume the same phase from its stable last checkpoint:

```bash
WANDB_RUN_NAME=resnet101-512x768-custom-anchor \
RESUME_LAST=1 EPOCHS=50 GPU=0 \
bash script.sh train
```

Or resume from an explicit checkpoint path:

```bash
WANDB_RUN_NAME=resnet101-512x768-custom-anchor \
RESUME_FROM=saved_results/resnet101-512x768-custom-anchor/checkpoints/last_model.pth \
EPOCHS=50 GPU=0 \
bash script.sh train
```

4. Predict and evaluate the same phase:

```bash
WANDB_RUN_NAME=baseline bash script.sh predict
WANDB_RUN_NAME=baseline bash script.sh evaluate
```

5. Hydra configuration:

```bash
python train_hydra.py
python predict_hydra.py
```

Hydra is configured to keep the current working directory unchanged. Training
logs/checkpoints still go to `saved_results/<run.name>/`.

Override parameters from the command line:

```bash
python train_hydra.py \
  run.name=resnet101-512x768-fliponly \
  model.backbone=resnet101 \
  model.min_size=512 \
  model.max_size=768 \
  augmentation.enabled=true \
  augmentation.horizontal_flip_probability=0.5 \
  augmentation.color_jitter_probability=0.0 \
  augmentation.grayscale_probability=0.0
```

The Hydra config also uses the custom detector:

```bash
python train_hydra.py \
  run.name=resnet101-custom \
  model.custom=true
```

Resume with Hydra:

```bash
python train_hydra.py \
  run.name=resnet101-512x768-fliponly \
  paths.resume_from=saved_results/resnet101-512x768-fliponly/checkpoints/last_model.pth \
  optim.epochs=50
```

Multi-GPU Hydra training can be requested with `device.gpus`. If it contains
more than one GPU id, DDP is launched automatically. The wrapper also sets the
canonical CUDA environment variable `CUDA_VISIBLE_DEVICES` for each train run:

```bash
python train_hydra.py \
  run.name=resnet101-ddp \
  device.gpus=0,1
```

The shell script does the same for regular training:

```bash
GPUS=0,1 WANDB_RUN_NAME=resnet101-ddp bash script.sh train
```
