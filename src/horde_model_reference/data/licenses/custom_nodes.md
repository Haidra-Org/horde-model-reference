# Custom Nodes - License Reference

**Last audited:** 2026-06-29

This file covers ComfyUI itself (vendored), bundled/vendored custom nodes, and nodes cloned from external repositories at install time.

---

## Vendored: ComfyUI

| Field | Value |
|---|---|
| **Name** | ComfyUI |
| **Location** | `ComfyUI/` (repo root of hordelib) |
| **Source** | https://github.com/comfyanonymous/ComfyUI |
| **Pinned commit** | `fb991e2c` (via `hordelib/installation/manifest.json`) |
| **License** | GPL v3 |
| **SPDX** | `GPL-3.0-only` |
| **Commercial use** | Yes - with copyleft obligations |
| **Notes** | Vendored in full. GPL v3 is compatible with hordelib's AGPL v3 top-level license. Any modifications to ComfyUI code within hordelib must remain GPL v3 or later. Distribution of hordelib (including ComfyUI) triggers copyleft obligations. |

---

## Vendored: comfyui_layerdiffuse

| Field | Value |
|---|---|
| **Name** | comfyui_layerdiffuse |
| **Location** | `hordelib/nodes/comfyui_layerdiffuse/` |
| **Source** | https://github.com/huchenlei/ComfyUI-layerdiffuse |
| **License** | Apache 2.0 |
| **SPDX** | `Apache-2.0` |
| **License file** | `hordelib/nodes/comfyui_layerdiffuse/LICENSE` |
| **Commercial use** | Yes |
| **Notes** | Vendored directly. Downloads LayerDiffusion model weights at runtime from `huggingface.co/LayerDiffusion/layerdiffusion-v1` (Apache 2.0). See `models.md` for weight-level details. |

---

## Vendored: facerestore_cf

| Field | Value |
|---|---|
| **Name** | facerestore_cf |
| **Location** | `hordelib/nodes/facerestore_cf/` |
| **Source** | https://github.com/jokersoul/comfyui-facerestore_cf (ComfyUI port of CodeFormer face restoration) |
| **License** | GPL v3 |
| **SPDX** | `GPL-3.0-only` |
| **License file** | `hordelib/nodes/facerestore_cf/LICENSE` |
| **Commercial use** | Yes - with copyleft obligations |
| **Notes** | This node is GPL v3 (the node code itself). However, it loads the CodeFormers and GFPGAN model weights at runtime. CodeFormers weights carry the S-Lab License 1.0 (non-commercial). See `models.md`. The copyleft-clean path for commercial use is to load only GFPGANv1.4 (Apache 2.0) and disable CodeFormers. |

---

## Cloned at install: comfyui_controlnet_aux

| Field | Value |
|---|---|
| **Name** | comfyui_controlnet_aux |
| **Location** | Cloned into custom nodes directory at install time |
| **Source** | https://github.com/Fannovel16/comfyui_controlnet_aux |
| **Pinned commit** | `e8b689a` (via `hordelib/installation/manifest.json`) |
| **License** | Apache 2.0 |
| **SPDX** | `Apache-2.0` |
| **Commercial use** | Yes |
| **Notes** | Provides ControlNet preprocessor/annotator nodes. Downloads annotator model weights from `huggingface.co/lllyasviel/Annotators` at runtime. Those weights have no license text - see the attention item in `README.md` and the entry in `models.md`. |

---

## Cloned at install: ComfyQR

| Field | Value |
|---|---|
| **Name** | ComfyQR |
| **Location** | Cloned into custom nodes directory at install time |
| **Source** | https://github.com/coreyryanhanson/ComfyQR |
| **Pinned commit** | `e31449c` (via `hordelib/installation/manifest.json`) |
| **License** | MIT |
| **SPDX** | `MIT` |
| **Commercial use** | Yes |
| **Notes** | No model weights involved. Pure node for QR code generation within ComfyUI pipelines. |

---

## hordelib-authored nodes

| Field | Value |
|---|---|
| **Name** | node_upscale_model_loader, node_controlnet_model_loader, node_image_loader, node_image_output, node_lora_loader, node_model_loader, and others |
| **Location** | `hordelib/nodes/*.py` |
| **License** | AGPL v3 |
| **SPDX** | `AGPL-3.0-only` |
| **Commercial use** | Yes - with copyleft obligations |
| **Notes** | Written by hordelib maintainers. Governed by the top-level hordelib license. |
