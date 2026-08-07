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
   p2pnet
   point-localization
   pytorch
   uav
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

Before upload, replace `owaisasghar/p2pnet-visdrone-head-counting` if the Hub
namespace differs. The contents of `huggingface/` are intended to become the
root of a Hugging Face model repository.

The model card uses a custom academic-only license. Do not change it to MIT,
Apache, or another permissive license.

## Hugging Face Space

The contents of `huggingface_space/` are intended to become a separate Gradio
Space. The Space downloads `model.safetensors` and `p2pnet_runtime.py` from the
model repository. Update `P2PNET_MODEL_ID` in the Space settings if the final
model ID differs.

Do not include VisDrone images as examples unless their redistribution terms
have been reviewed. Prefer an authorized image from the intended deployment
camera, and label any external-domain result clearly.

## Claims checklist

- Say “P2PNet adaptation” or “extension,” not “reproduction of the MDPI model.”
- Report MAE/RMSE for counts and pixel-radius P/R/F1/AP for localization.
- State that the current metrics use one validation fold.
- Do not call 4.11 full-resolution forward FPS “real time.”
- Do not claim state of the art without a matched independent benchmark.
