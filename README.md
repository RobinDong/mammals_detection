# mammals_detection
Detect mammals in pictures

## Screen texture training (v2)

`ScratchTrainer` applies image-only augmentation to 35% of training images across
all classes. About 21.9% receive crossing screen grids with varying pitch and RGB
phases; about 4.4% each receive Gaussian blur, resampling or JPEG compression.
65% pass through this stage unchanged. Boxes/classes are preserved and the stage
survives closing mosaic. `SCREEN_AUGMENTATION_P` controls the overall probability.

`train.py` uses a base size of 896 with `multi_scale=1/7`, sampling stride-aligned
batch sizes from 768 to 1024. This also varies the scale of the synthetic texture.
A running process must be restarted to use changed code.

To fine-tune the completed checkpoint rather than start from random weights:

```bash
conda activate yolo
cd /home/robin/code/mammals_detection
python3 train.py --weights runs/detect/yolo26l_finetune_768_texture_100e/weights/best.pt --epochs 30
```

On robin4090, use the `yolo` environment (Ultralytics 8.4.62, Python 3.10).
The Conda base environment has Ultralytics 8.3.202 and does not accept `cls_pw`.
The entry point checks this before loading weights. If activation is unavailable,
use `/home/robin/miniconda3/envs/yolo/bin/python` in place of `python3`.

This starts a separate `yolo26l_finetune_texture_v2` run with AdamW and cosine LR
1e-4 to 1e-5, including one warmup epoch. `python3 train.py` without `--weights`
still starts from scratch. `resume.py` retains its original explicit size and
schedule; use the command above for the v2 multiscale settings.

## Validation

Full clean validation and standard `best.pt` selection remain based on mAP.
Each epoch also evaluates a fixed class-balanced subset of validation images
(up to 16 per class) at 768 and 1024, both clean and with seeded screen grids.
The probe uses square inputs to cover fixed-shape export deployment.

The probe reports **localized top-class recall**, not mAP: a ground-truth object
counts only when its highest-scoring overlapping prediction (IoU >= 0.5,
confidence >= 0.25) has the correct class. A higher-scoring wrong class therefore
cannot be hidden by another label on the same box. `probe/score` averages the
macro recall over the two sizes and clean/corrupt conditions. `best_texture.pt`
is an additional checkpoint selected by this score. Compare its clean mAP too;
the small synthetic probe does not establish real-world generalization.

`texture_probe_manifest.json` records the selected validation images;
`texture_probe.jsonl` and `results.csv` record metrics. No supplied failure photo
is added to training or checkpoint selection. Normal validation and inference
images are not smoothed or otherwise changed by this probe.

Run integration tests in the yolo environment:

```bash
python3 -m unittest discover -s tests -v
```

## Accuracy-checked INT8 ONNX export

```bash
conda activate yolo
python3 export.py
python3 test.py
```

The export defaults to `yolo26l_finetune_texture_v2-2/weights/best.pt` and its
stored image size (896). Pass `--model` and `--imgsz` to change them. The output
is a fixed square, batch-1 ONNX. Calibration now uses 512 deterministic,
class-stratified **training** images, separate from validation. The first five
YOLO modules (0–4) and the Detect head remain FP32; the other convolution
weights/activations use per-channel-weight U8/S8 QDQ quantization with MinMax
calibration. This is a mixed-precision INT8 model, not an all-INT8 network.

Before replacing an output, the exporter runs the FP32 and INT8 ONNX models on
the **same complete validation split**, with identical square inputs, batch 1,
FP32 input tensors and detection thresholds. It rejects exports losing more
than 0.003 overall mAP50–95, or 0.005 in any class's mAP50 or mAP50–95 (absolute
units: 0.005 means 0.5 percentage points). A failure leaves the previous output
untouched. `*.accuracy.json` records model hashes, calibration image paths,
validation settings and accuracy results; rejected runs save
`*.rejected.accuracy.json`. Full validation takes several minutes on the 4090.
`--skip-map-check` explicitly bypasses this check and marks the report as
accuracy-unverified. GPU calibration/validation is selected automatically when
available; use `--device cpu` for CPU execution.

`test.py` now uses `robin_extra_seahorse.yaml`, matching training's 18,981-image
validation set, with `rect=False` and `half=False`. Compare PT and ONNX with
these same settings: training's rectangular validation and a fixed square
ONNX input are different evaluation conditions. These accuracy checks measure
this validation distribution, not robustness to every new photo.
