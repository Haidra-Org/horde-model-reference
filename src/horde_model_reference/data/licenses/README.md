# hordelib Dependency Licenses

**Last audited:** 2026-06-29
**Scope:** hordelib (including vendored ComfyUI and bundled custom nodes) and all ML model weights referenced or downloaded at runtime.

This directory documents the provenance and license status of every model weight and ComfyUI custom node that hordelib uses. It is intended as a living reference for maintainers evaluating deployment constraints, particularly for commercial or redistributed use.

---

## Files in this directory

| File | Contents |
|---|---|
| `custom_nodes.md` | ComfyUI custom nodes (vendored, cloned at install, and hordelib-authored) |
| `models.md` | All ML model weights (ControlNet, upscalers, face restoration, CLIP, etc.) |

---

## Methodology

This audit was conducted by:

1. Statically analysing the hordelib source tree: `hordelib/nodes/`, pipeline JSON files, model database JSON files (`hordelib/model_database/`), and the install manifest (`hordelib/installation/manifest.json`).
2. Cross-referencing `horde-model-reference` annotator catalog and model category files.
3. Fetching HuggingFace model card metadata and GitHub repository license fields for any item whose license was not co-located in the source tree.
4. Consulting openmodeldb.info for upscaler model provenance where HuggingFace records were absent.

License classifications use the SPDX identifier where a standard license applies. Items tagged **"other"** or **"unknown"** have no machine-readable license declaration at their source.

---

## Items Requiring Attention

The following items have licenses that restrict commercial use or redistribute under ambiguous terms. Any deployment of hordelib that is commercial or involves redistribution of weights should review these before proceeding.

### 1. `thibaud/controlnet-sd21` - Custom restricted license

- **Affects:** All SD 2.1 ControlNet variants: canny, depth, hed, openpose, scribble, fakescribbles (fp21 weights).
- **License:** HuggingFace metadata tag "other"; the model card contains an explicit custom restriction: *"Models can't be sold, merged, or distributed without prior written agreement."*
- **Risk:** High. The restriction on distribution is unusually broad and covers redistribution in bundled or pre-downloaded form (e.g., Docker images, worker packages). Written permission from the author (thibaud) is required before redistribution.
- **Action options:** (a) obtain written agreement from the author, (b) substitute with the `lllyasviel/ControlNet` SD2.1 weights (Apache 2.0), or (c) disable SD2.1 ControlNet support in any commercial/redistributed build.

### 2. `4x_AnimeSharp` - CC BY-NC-SA 4.0 (non-commercial)

- **Affects:** The `4x_AnimeSharp` upscaler model served via `esrgan.json`.
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International.
- **Risk:** Medium-High. NC clause prohibits commercial use. ShareAlike requires any derivative works to use the same license.
- **Action options:** (a) remove from the default model list for commercial deployments, (b) replace with a commercially-licensed upscaler (e.g., RealESRGAN_x4plus under BSD 3-Clause).

### 3. `RealESRGAN_x2plus` and `CodeFormers` - S-Lab License 1.0 (non-commercial research only)

- **Affects:** The `RealESRGAN_x2plus` upscaler model and the `CodeFormers` face restoration model.
- **License:** S-Lab License 1.0. Explicit terms: *"The Code and Model can be used for non-commercial research purposes only."*
- **Risk:** Medium-High. Both models are from `sczhou/CodeFormer`. Any commercial service offering upscaling via `x2plus` or CodeFormer face restoration is in breach of this license.
- **Action options:** (a) restrict these features to non-commercial tiers, (b) remove from commercial builds and substitute a BSD/Apache-licensed alternative for upscaling (RealESRGAN_x4plus is BSD 3-Clause).

### 4. `lllyasviel/Annotators` - No license text

- **Affects:** All ControlNet preprocessor/annotator weights: `ControlNetHED.pth`, `res101.pth` (LeReS), `latest_net_G.pth` (LeReS), `body_pose_model.pth`, `hand_pose_model.pth`, `facenet.pth` (OpenPose), `upernet_global_small.pth` (semantic segmentation), `mlsd_large_512_fp32.pth`.
- **License:** HuggingFace metadata tag "other"; no license text is present in the repository.
- **Risk:** Medium. These weights are community-uploaded derivatives of multiple research codebases, each carrying its own upstream terms (HED: MIT-adjacent research license; OpenPose: custom non-commercial for commercial use; LeReS: MIT; M-LSD: Apache 2.0; UperNet/ADE20K: research use). The absence of a unifying license at the HuggingFace repo means terms cannot be verified from that source alone.
- **Action options:** (a) audit each weight's upstream research paper and original code repository for license terms, (b) contact lllyasviel for clarification, (c) download weights directly from the upstream repositories where license terms are defined.

---

## Summary: Commercial use quick-reference

| Item | Commercial use permitted |
|---|---|
| hordelib (AGPL v3) | Yes - with copyleft obligations |
| ComfyUI (GPL v3) | Yes - with copyleft obligations |
| comfyui_layerdiffuse | Yes (Apache 2.0) |
| facerestore_cf node | Yes (GPL v3) |
| comfyui_controlnet_aux | Yes (Apache 2.0) |
| ComfyQR | Yes (MIT) |
| SD 1.5 inpainting / Safety checker | Yes - with RAIL-M use restrictions |
| CLIP models (OpenAI, LAION) | Yes (MIT) |
| kohya-ss SD1.5 ControlNet weights | Yes (Apache 2.0) |
| **thibaud SD2.1 ControlNet weights** | **No - written agreement required** |
| **lllyasviel/Annotators** | **Unknown - no license text** |
| NMKD_Siax, 4x_NMKD_Superscale_SP | Yes (WTFPL) |
| **4x_AnimeSharp** | **No (CC BY-NC-SA 4.0)** |
| RealESRGAN_x4plus, x4plus_anime_6B | Yes (BSD 3-Clause) |
| **RealESRGAN_x2plus** | **No (S-Lab License 1.0)** |
| GFPGANv1.4 | Yes (Apache 2.0) |
| facexlib detection/parsing weights | Yes (BSD 3-Clause) |
| **CodeFormers** | **No (S-Lab License 1.0)** |
| BLIP base / BLIP Large | Yes (BSD 3-Clause) |
| Qwen diffusion/encoder/VAE | Yes (Apache 2.0) |
| Qwen-Image-Lightning LoRA | Yes (Apache 2.0) |
| LayerDiffusion weights | Yes (Apache 2.0) |
