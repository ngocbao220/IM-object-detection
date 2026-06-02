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

The current model uses Faster R-CNN with a ResNet-101 FPN backbone. Detection
heads are initialized from scratch; only the optional backbone weights come
from ImageNet. Use a new phase name when comparing with older ResNet-50 runs:

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

Training artifacts are stored under:

```text
saved_results/<WANDB_RUN_NAME>/
├── checkpoints/
│   ├── best_model.pth
│   └── last_model.pth
└── logs/
    └── session.log
```

4. Predict and evaluate the same phase:

```bash
WANDB_RUN_NAME=baseline bash script.sh predict
WANDB_RUN_NAME=baseline bash script.sh evaluate
```
