# Qwen Image 2.1

Qwen Image 2.1 uses its own `QWEN_IMAGE_21` model type. It is not compatible with the older `QWEN` model type.

The edit-training preset expects each target image to have a paired conditioning image with the same base filename and
the `-condlabel` suffix. For example:

```text
sample.jpg
sample.txt
sample-condlabel.png
```

From the repository root, the included minimal smoke configuration can be run with:

```powershell
.\venv\Scripts\python.exe scripts\train.py `
  --preset-path "training_presets\QwenImage21\#qwen-image-2.1 Edit LoRA int8.json" `
  --config-path "docs\examples\qwen-image-21-minimal-edit.json"
```

The preset enables paired conditioning images, INT8 weights for the transformer and Qwen3-VL encoder, compilation,
dynamic timestep shifting using the checkpoint scheduler parameters, and the Qwen Image 2.1 transformer layer names.
Text-encoder training is not currently supported; the text/vision conditioning is cached before transformer training.

When `custom_conditioning_image` is enabled, the sample editor shows a dedicated **conditioning image path** field.
Qwen Image 2.1 passes that image to the Diffusers edit pipeline as both vision context and VAE conditioning. The
**base image** and **mask image** fields are reserved for masked inpainting models and are not used by Qwen Image 2.1.
