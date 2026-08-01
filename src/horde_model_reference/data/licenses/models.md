# Model Weights - License Reference

**Last audited:** 2026-06-29

This file covers every ML model weight that hordelib downloads or references at runtime, grouped by functional category. Items marked with a warning flag require attention for commercial or redistributed deployments - see `README.md` for full detail on each.

---

## Stable Diffusion / Core Diffusion Models

### SD 1.5 Inpainting

| Field | Value |
|---|---|
| **Name** | stable-diffusion-inpainting |
| **HuggingFace repo** | `runwayml/stable-diffusion-inpainting` |
| **Referenced in** | `hordelib/model_database/diffusers.json` |
| **License** | CreativeML Open RAIL-M |
| **Commercial use** | Yes - subject to RAIL-M use restrictions (no harmful content generation, etc.) |
| **Notes** | The RAIL-M license is a Responsible AI license. It does not restrict commercial use but attaches behavioural use-case restrictions that propagate to downstream deployments. |

### Safety Checker

| Field | Value |
|---|---|
| **Name** | stable-diffusion-safety-checker |
| **HuggingFace repo** | `CompVis/stable-diffusion-safety-checker` |
| **Referenced in** | `hordelib/model_database/safety_checker.json` |
| **License** | CreativeML Open RAIL-M |
| **Commercial use** | Yes - subject to RAIL-M use restrictions |
| **Notes** | Same RAIL-M terms as above. |

---

## CLIP Interrogation Models

All CLIP models are MIT licensed and permit commercial use without restriction.

| Name | Source | HuggingFace / URL | License |
|---|---|---|---|
| ViT-L/14 | OpenAI | `openaipublic.azureedge.net` | MIT |
| coca_ViT-L-14 (mscoco fine-tuned) | LAION | `laion/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k` | MIT |
| ViT-bigG-14 | LAION | `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` | MIT |
| ViT-g-14 | LAION | `laion/CLIP-ViT-g-14-laion2B-s12B-b42K` | MIT |
| ViT-H-14 | LAION | `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | MIT |

**Referenced in:** `hordelib/model_database/clip.json`

---

## ControlNet Models (SD 1.5)

All SD 1.5 ControlNet weights are from `kohya-ss/ControlNet-diff-modules` (Apache 2.0).

| Filename | Control type | License | Commercial use |
|---|---|---|---|
| `diff_control_sd15_canny_fp16.safetensors` | canny | Apache 2.0 | Yes |
| `diff_control_sd15_depth_fp16.safetensors` | depth | Apache 2.0 | Yes |
| `diff_control_sd15_hed_fp16.safetensors` | hed | Apache 2.0 | Yes |
| `diff_control_sd15_mlsd_fp16.safetensors` | mlsd, hough | Apache 2.0 | Yes |
| `control_normal_fp16.safetensors` | normal | Apache 2.0 | Yes |
| `control_openpose_fp16.safetensors` | openpose | Apache 2.0 | Yes |
| `control_scribble_fp16.safetensors` | scribble, fakescribbles | Apache 2.0 | Yes |
| `control_seg_fp16.safetensors` | seg | Apache 2.0 | Yes |

**HuggingFace repo:** `kohya-ss/ControlNet-diff-modules`
**Referenced in:** `hordelib/model_database/controlnet.json`

---

## ControlNet Models (SD 2.1) - ATTENTION REQUIRED

> **Warning:** These weights carry a custom restricted license. See the attention item in `README.md` before using in any commercial or redistributed context.

| Filename | Control type | License | Commercial use |
|---|---|---|---|
| `diff_control_sd21_canny_fp16.safetensors` | canny | Other (custom) | **No** |
| `diff_control_sd21_depth_fp16.safetensors` | depth | Other (custom) | **No** |
| `diff_control_sd21_hed_fp16.safetensors` | hed | Other (custom) | **No** |
| `control_openpose_fp21.safetensors` | openpose | Other (custom) | **No** |
| `control_scribble_fp21.safetensors` | scribble, fakescribbles | Other (custom) | **No** |

**HuggingFace repo:** `thibaud/controlnet-sd21`
**License detail:** Model card states: *"Models can't be sold, merged, or distributed without prior written agreement."*
**Referenced in:** `hordelib/model_database/controlnet.json`

---

## ControlNet Annotator (Preprocessor) Weights - ATTENTION REQUIRED

> **Warning:** No license text is present at the HuggingFace source (`lllyasviel/Annotators`). Each weight is a derivative of a different upstream research codebase with its own terms. See the attention item in `README.md`.

These weights are downloaded at runtime by `comfyui_controlnet_aux`.

| Filename | Used by | Upstream research source | Known upstream license |
|---|---|---|---|
| `ControlNetHED.pth` | HED, fakescribbles preprocessor | Holistically-nested Edge Detection (BSDS) | Research - not formally open |
| `res101.pth` | LeReS depth preprocessor | LeReS (SJTU) | MIT |
| `latest_net_G.pth` | LeReS depth preprocessor | LeReS (SJTU) | MIT |
| `body_pose_model.pth` | OpenPose body preprocessor | OpenPose (CMU) | Custom - non-commercial for commercial |
| `hand_pose_model.pth` | OpenPose hand preprocessor | OpenPose (CMU) | Custom - non-commercial for commercial |
| `facenet.pth` | OpenPose face preprocessor | OpenPose (CMU) | Custom - non-commercial for commercial |
| `upernet_global_small.pth` | Semantic segmentation preprocessor | UperNet / ADE20K | Research use |
| `mlsd_large_512_fp32.pth` | M-LSD / hough preprocessor | M-LSD (NAVER) | Apache 2.0 |
| `sk_model.pth` | LineArt (realistic) preprocessor | Informative Drawings (`carolineec/informative-drawings`) | MIT |
| `sk_model2.pth` | LineArt (coarse-mode weight) preprocessor | Informative Drawings (`carolineec/informative-drawings`) | MIT |
| `netG.pth` | Anime LineArt preprocessor | Anime2Sketch (`Mukosame/Anime2Sketch`) | MIT |
| `erika.pth` | Manga (anime denoise) LineArt preprocessor | MangaLineExtraction (`ljsabc/MangaLineExtraction_PyTorch`) | MIT |
| `table5_pidinet.pth` | PiDiNet soft-edge / scribble preprocessor | PiDiNet (`hellozhuo/pidinet`) | Research - no explicit SPDX at upstream |
| `scannet.pt` | BAE normal-map preprocessor | Bae et al. surface-normal (`baegwangbin/surface_normal_uncertainty`) | MIT |

**HuggingFace repo:** `huggingface.co/lllyasviel/Annotators` (community upload, no unified license)
**HuggingFace API `cardData.license`:** `other` (confirmed 2026-07-11)
**Referenced via:** `comfyui_controlnet_aux` node, `horde-model-reference` annotator catalog

---

## ControlNet Annotator (Preprocessor) Weights - NON-COMMERCIAL, DO NOT MIRROR

> **Warning:** These two annotator weights carry explicit NonCommercial licenses on their HuggingFace source
> repos. They are deliberately **excluded** from the gated R2 redistribution allowlist
> (`scripts/r2_sync/redistribution_policy.json`): the worker fetches them straight from the upstream
> HuggingFace origin instead, and no copy is hosted on the mirror. Do not approve them in the policy without a
> license reassessment.

| Filename | Used by | HuggingFace source | License (confirmed via HF API) | Commercial use |
|---|---|---|---|---|
| `7_model.pth` | TEED soft-edge preprocessor | `bdsqlsz/qinglong_controlnet-lllite` (subfolder `Annotators`) | CC BY-NC-SA 4.0 | **No** |
| `depth_anything_v2_vitl.pth` | Depth-Anything-V2 depth preprocessor | `depth-anything/Depth-Anything-V2-Large` | CC BY-NC-4.0 | **No** |

**TEED note:** the upstream TEED research code (`xavysp/TEED`) is MIT, but the weight is fetched from the
`bdsqlsz/qinglong_controlnet-lllite` repo, whose HF `cardData.license` is `cc-by-nc-sa-4.0`. Redistribution from
that source is therefore NonCommercial + ShareAlike, so the file is left on origin and off the R2 mirror.

**Depth-Anything-V2 note:** the **Large** checkpoint (`depth_anything_v2_vitl.pth`) is CC BY-NC-4.0
(NonCommercial). The Depth-Anything-V2 **Small** and **Base** checkpoints are Apache-2.0. If a commercially
redistributable depth-anything annotator is required, swap the catalog entry to the Small/Base Apache-2.0
variant; otherwise this weight remains fetched from upstream only and is not mirrored.

---

## Upscaler Models (ESRGAN)

### Commercially safe

| Name | Source | License | SPDX | Commercial use |
|---|---|---|---|---|
| RealESRGAN_x4plus | `xinntao/Real-ESRGAN` GitHub releases | BSD 3-Clause | `BSD-3-Clause` | Yes |
| RealESRGAN_x4plus_anime_6B | same | BSD 3-Clause | `BSD-3-Clause` | Yes |
| NMKD_Siax (`4x_NMKD-Siax_200k`) | openmodeldb.info / nmkd.de | WTFPL | `WTFPL` | Yes |
| 4x_NMKD_Superscale_SP | openmodeldb.info / nmkd.de | WTFPL | `WTFPL` | Yes |
| 4xNomos8kSC (ESRGAN, 4x) | `Phhofm/models` GitHub | CC BY 4.0 | `CC-BY-4.0` | Yes (attribution) |
| 4xLSDIRplus (ESRGAN, 4x) | `Phhofm/models` GitHub | CC BY 4.0 | `CC-BY-4.0` | Yes (attribution) |
| 4xNomosWebPhoto_RealPLKSR (RealPLKSR, 4x) | `Phhofm/models` GitHub | CC BY 4.0 | `CC-BY-4.0` | Yes (attribution) |
| 4xNomos2_realplksr_dysample (RealPLKSR, 4x) | `Phhofm/models` GitHub | CC BY 4.0 | `CC-BY-4.0` | Yes (attribution) |
| 4xNomos2_hq_dat2 (DAT2, 4x) | `Phhofm/models` GitHub | CC BY 4.0 | `CC-BY-4.0` | Yes (attribution) |
| 2xModernSpanimationV1 (SPAN, 2x) | `TNTwise/Models` GitHub | MIT | `MIT` | Yes |

The six entries above are a modern, permissively-licensed batch loaded via spandrel's core registry.
They are distributed through the horde-model-reference PRIMARY service (`models.aihorde.net`) pending
queue as beta models, **not** the GitHub `esrgan.json` (older workers cannot load the newer
architectures). The CC-BY-4.0 entries require attribution to their author (Philip Hofmann / Phhofm).

### Non-commercial - ATTENTION REQUIRED

| Name | Source | License | Commercial use | Notes |
|---|---|---|---|---|
| RealESRGAN_x2plus | `sczhou/CodeFormer` GitHub releases | S-Lab License 1.0 | **No** | Non-commercial research only. Despite being hosted in the CodeFormer repo, this weight is a standalone ESRGAN model. |
| 4x_AnimeSharp | `Kim2091/AnimeSharp` on HuggingFace | CC BY-NC-SA 4.0 | **No** | NonCommercial + ShareAlike. Attribution required. |

**Referenced in:** `hordelib/model_database/esrgan.json`

---

## Face Restoration Models

### GFPGANv1.4

| Field | Value |
|---|---|
| **Name** | GFPGANv1.4 (`GFPGANv1.4.pth`) |
| **Source** | GitHub releases: `TencentARC/GFPGAN` |
| **License** | Apache 2.0 |
| **SPDX** | `Apache-2.0` |
| **Commercial use** | Yes |
| **Notes** | Loaded by the `facerestore_cf` node. |

### GFPGANv1.3

| Field | Value |
|---|---|
| **Name** | GFPGANv1.3 (`GFPGANv1.3.pth`) |
| **Source** | GitHub releases: `TencentARC/GFPGAN` (`v1.3.0`) |
| **License** | Apache 2.0 |
| **SPDX** | `Apache-2.0` |
| **Commercial use** | Yes |
| **Notes** | The predecessor GFPGAN weight, more identity-faithful than v1.4. Same StyleGAN2 arch, loaded by the `facerestore_cf` node's GFPGAN path. Served as beta via the `gfpgan` category pending queue. |

### RestoreFormer

| Field | Value |
|---|---|
| **Name** | RestoreFormer (`RestoreFormer.pth`) |
| **Source** | GitHub releases: `TencentARC/GFPGAN` (`v1.3.4`); arch from `wzhouxiff/RestoreFormer` (Apache 2.0) |
| **License** | Apache 2.0 |
| **SPDX** | `Apache-2.0` |
| **Commercial use** | Yes |
| **Notes** | Transformer/VQGAN blind face restorer (CVPR 2022), realism-oriented. Detected and loaded through spandrel's core registry rather than the GFPGAN loader; catalogued in the `gfpgan` category (which maps to the same `facerestore_models` folder). Served as beta via the pending queue. |

### facexlib detection and parsing weights

| Name | Filename | Source | License | Commercial use |
|---|---|---|---|---|
| Face detection | `detection_Resnet50_Final.pth` | `xinntao/facexlib` GitHub | BSD 3-Clause | Yes |
| Face parsing | `parsing_parsenet.pth` | `xinntao/facexlib` GitHub | BSD 3-Clause | Yes |

Both are dependencies of GFPGANv1.4 and downloaded automatically via the `facexlib` Python package.

### CodeFormers - ATTENTION REQUIRED

| Field | Value |
|---|---|
| **Name** | CodeFormers (`CodeFormers.pth`) |
| **Source** | GitHub releases: `sczhou/CodeFormer` |
| **License** | S-Lab License 1.0 |
| **Commercial use** | **No** |
| **Notes** | Explicit terms: *"The Code and Model can be used for non-commercial research purposes only."* Loaded by the `facerestore_cf` node alongside GFPGANv1.4. Commercial deployments should configure the node to use GFPGAN only. |

---

## BLIP Captioning Models

| Name | Source | License | SPDX | Commercial use |
|---|---|---|---|---|
| BLIP base | Salesforce GCS (`sfr-vision-language-research/BLIP`) | BSD 3-Clause | `BSD-3-Clause` | Yes |
| BLIP Large | same | BSD 3-Clause | `BSD-3-Clause` | Yes |

**Referenced in:** `hordelib/model_database/blip.json`

---

## Qwen Image Generation Models

All Qwen pipeline weights are Apache 2.0 and permit commercial use.

| Name | Role in pipeline | HuggingFace repo | License | Commercial use |
|---|---|---|---|---|
| `qwen_image_fp8_e4m3fn.safetensors` | Diffusion model | `Comfy-Org/Qwen-Image_ComfyUI` | Apache 2.0 | Yes |
| `qwen_2.5_vl_7b_fp8_scaled.safetensors` | Text encoder | `Comfy-Org/Qwen-Image_ComfyUI` | Apache 2.0 | Yes |
| `qwen_image_vae.safetensors` | VAE | `Comfy-Org/Qwen-Image_ComfyUI` | Apache 2.0 | Yes |
| `Qwen-Image-Lightning-8steps-V1.0.safetensors` | Lightning LoRA (speed) | `lightx2v/Qwen-Image-Lightning` | Apache 2.0 | Yes |

**Referenced in:** `hordelib/pipeline_designs/pipeline_qwen.json`
**Notes:** The Qwen 2.5 base model license is the Qwen Research License, which Comfy-Org republishes under Apache 2.0 for the ComfyUI-packaged versions. The `lightx2v/Qwen-Image-Lightning` repo declares Apache 2.0 explicitly on HuggingFace.

---

## LayerDiffusion Weights

All LayerDiffusion weights are Apache 2.0 and permit commercial use.

**HuggingFace repo:** `LayerDiffusion/layerdiffusion-v1`
**Downloaded by:** `comfyui_layerdiffuse` node at runtime
**Referenced in:** `hordelib/pipeline_designs/` (pipelines using layerdiffuse)

| Filename | Use case |
|---|---|
| `layer_sd15_vae_transparent_decoder.safetensors` | SD 1.5 transparent VAE decoder |
| `vae_transparent_decoder.safetensors` | SDXL transparent VAE decoder |
| `layer_xl_transparent_attn.safetensors` | SDXL attention-based transparency |
| `layer_xl_transparent_conv.safetensors` | SDXL conv-based transparency |
| `layer_sd15_transparent_attn.safetensors` | SD 1.5 attention-based transparency |
| `layer_sd15_joint.safetensors` | SD 1.5 joint foreground+background generation |
| `layer_xl_fg2ble.safetensors` | SDXL foreground to blended |
| `layer_xl_bg2ble.safetensors` | SDXL background to blended |
| `layer_sd15_fg2bg.safetensors` | SD 1.5 foreground to background |
| `layer_sd15_bg2fg.safetensors` | SD 1.5 background to foreground |
| `layer_xl_fgble2bg.safetensors` | SDXL blended-foreground to background |
| `layer_xl_bgble2fg.safetensors` | SDXL blended-background to foreground |

---

## Community Embeddings / Textual Inversions

These are downloaded from the AI-Horde GitHub releases and managed via `db_embeds.json`. They are community-contributed and largely lack formal licenses. They are optional quality-of-life additions (negative prompts, style embeddings) and do not carry the same commercial risk as core model weights since they are not required for operation.

| Name | Purpose | License |
|---|---|---|
| bad_prompt | Negative quality embedding | Community - no formal license |
| EasyNegative | Negative quality embedding | Community - no formal license |
| nfixer | Negative quality embedding | Community - no formal license |
| epiNoiseOffset_v2 | Noise offset style embedding | Community - no formal license |
| HyperSmoke | Style embedding | Community - no formal license |
