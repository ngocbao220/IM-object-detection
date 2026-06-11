docker build -t object-detection-exam:2026 .

# cd my_submission

mkdir -p grading_outputs

docker run --rm --gpus all \
  -v "$PWD/public/val/images:/exam/val_images:ro" \
  -v "$PWD:/workspace" \
  -v "$PWD/grading_outputs:/exam/outputs" \
  object-detection-exam:2026 \
  python predict.py \
    --image_dir /exam/val_images \
    --output /exam/outputs/val_predictions.json

python public/tools/evaluate_predictions.py \
  --ground_truth public/annotations/val.json \
  --predictions grading_outputs/val_predictions.json \
  --output grading_outputs/val_score.json