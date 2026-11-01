# Portfolio publishing checklist

This checklist prepares the project for GitHub and Hugging Face without
performing any remote write automatically.

## GitHub

1. Review all local changes and confirm that no dataset, private video, token,
   or training checkpoint is tracked.
2. Confirm that publishing the repository and inference checkpoint complies
   with the P2PNet academic-research-only license.
3. Make the showcase repository public, or create a separate public case-study
   repository if the implementation must remain private.
4. Add these repository topics:

   ```text
   aerial-imagery
   computer-vision
   crowd-counting
   head-detection
   multi-object-tracking
   p2pnet
   point-localization
   pytorch
   uav
   uav-dot
   visdrone
   ```

5. Upload `assets/uav_p2pnet_social_preview.jpg` as the repository social
   preview.
6. Publish the model release only after its source tag matches the evaluated
   code and the checkpoint checksum in `MODEL_CARD.md`.
7. Pin the public repository on the GitHub profile and link its Hugging Face
   model and Space.
8. Replace the one-byte profile README with the reviewed template in
   `docs/GITHUB_PROFILE_README.md`, then add only verified contact and social
   links.

## Hugging Face model repository

Prepare the inference-only weights locally:

```bash
PYTHONPATH=. .venv/bin/python scripts/export_huggingface_model.py \
  --checkpoint runs/p2pnet_visdrone_fold0/checkpoints/best_mae.pth \
  --output huggingface/model.safetensors
```

The Hugging Face account is `uwaisasghar` (GitHub: `owaisasghar`), so the model
repository is `uwaisasghar/p2pnet-visdrone-head-counting`. The contents of `huggingface/` are intended to become the
root of a Hugging Face model repository.

The model card uses a custom academic-only license. Do not change it to MIT,
Apache, or another permissive license.

## Hugging Face Space

Hosting Gradio Spaces, even on the free CPU tier, requires a Hugging Face PRO
subscription; creating one without PRO fails with `402 Payment Required`. Until
then, the free public demo is `notebooks/demo.ipynb` (README "Open in Colab"
badge), which runs the same models.

The contents of `huggingface_space/` are intended to become a separate Gradio
Space that compares UAV-Dot and P2PNet side by side:

- **P2PNet:** the Space downloads `model.safetensors` and `p2pnet_runtime.py`
  from the model repository. Update `P2PNET_MODEL_ID` in the Space settings if
  the final model ID differs.
- **UAV-Dot:** `uavdot_runtime.py` fetches the released weights from the UAV-Dot
  authors' public mirror on first use and checks their SHA-256, so the Space
  never re-hosts them. Keep `LICENSE-uav-dot.txt` with the runtime (Apache-2.0).

Test it locally before pushing:

```bash
cd huggingface_space
P2PNET_MODEL_DIR=../huggingface ../.venv/bin/python app.py
```

Do not include VisDrone images as examples unless their redistribution terms
have been reviewed. Prefer an authorized image from the intended deployment
camera, and label any external-domain result clearly.

## Claims checklist

- Say “P2PNet adaptation” or “extension,” not “reproduction of the MDPI model.”
- Report MAE/RMSE for counts and pixel-radius P/R/F1/AP for localization.
- State that the validation metrics use one fold; quote test-set numbers
  (`runs/eval/test_metrics.json`) as the headline.
- Speed claims: UAV-Dot 29 ms (34 FPS) and P2PNet 115 ms per 1080p frame are
  model-only fp16 timings on an RTX 4060 Laptop GPU; end-to-end with decoding is
  42 ms and 128 ms. Quote the hardware with every number.
- UAV-Dot's weights and the tracker are third-party (Apache-2.0): say “integrated,
  evaluated and optimized,” never “developed.” The UP-COUNT fine-tune did not beat
  the released weights on test; present it as a negative result.
- Do not claim state of the art without a matched independent benchmark.
