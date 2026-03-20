# PoseBench: Virtual Try-On Model Exploration

> **Goal:** Explore and compare state-of-the-art diffusion-based virtual try-on models (CatVTON, IDM-VTON etc) to understand their practical performance on diverse poses and in-the-wild images.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/Platform-Kaggle-20BEFF)](https://www.kaggle.com/)

## 🎯 Project Overview

This project explores recent diffusion-based virtual try-on models, focusing on their robustness beyond standard benchmark scenarios. Key areas of investigation:
- Performance on non-frontal poses (lateral, rear views)
- Handling of in-the-wild images with complex backgrounds
- Garment detail preservation across different pattern complexities
- Comparison of different architectural approaches (spatial concatenation vs. dual-encoder)

**Current Status:** Model implementation and initial testing phase - comparing CatVTON and IDM-VTON architectures.

## 🔬 Models Implemented

### 1. CatVTON (ICLR 2025)
**Notebook:** `catvton2.ipynb`  
**Architecture:** Single UNet with spatial concatenation  
**Key Features:**
- Lightweight (899M total params, 49M trainable)
- Efficient inference (<8GB VRAM)
- Novel spatial concatenation approach

**Status:** ✅ Successfully implemented and tested

### 2. IDM-VTON (2024)
**Notebook:** `virtual-try-on-4.ipynb`  
**Architecture:** Dual-encoder (GarmentNet + IP-Adapter)  
**Key Features:**
- High-level semantic + low-level detail fusion
- Based on Stable Diffusion XL
- Strong detail preservation

**Status:** ✅ Successfully implemented with custom patches

## 📂 Repository Structure
```
├── catvton2.ipynb              # CatVTON implementation & inference
├── virtual-try-on-4.ipynb      # IDM-VTON implementation & inference  
├── test_images/                # Custom test images
│   ├── person/
│   └── garment/
├── results/                    # Generated try-on results
│   ├── catvton/
│   └── idm_vton/
└── README.md
```

## 🚀 Getting Started

### Platform
Both notebooks are designed to run on **Kaggle** with:
- GPU: T4 or P100 (16GB VRAM)
- Datasets: VITON-HD, DressCode (via Kaggle datasets)
- Internet enabled for model downloads

### Running the Notebooks

#### CatVTON (`catvton2.ipynb`)
1. Upload notebook to Kaggle
2. Enable GPU accelerator
3. Add VITON-HD dataset (if testing on benchmark)
4. Run cells sequentially
5. Results saved to `/kaggle/working/results/`

#### IDM-VTON (`virtual-try-on-4.ipynb`)
1. Upload notebook to Kaggle
2. Enable GPU + Internet
3. Set up Kaggle secrets for GitHub token (if using private repo)
4. Add VITON-HD dataset from Kaggle datasets
5. Run cells - includes automatic patching for compatibility
6. Results visualized in comparison grids

### Key Implementation Details

**CatVTON:**
- Direct inference pipeline
- Minimal preprocessing required
- Fast iteration for testing

**IDM-VTON:**
- Requires compatibility patches (diffusers version fixes)
- Custom preprocessing for mask handling
- Memory optimization for Kaggle's VRAM limits
- Includes visual inspection tools for pair selection

## 📊 Initial Observations

### Architectural Comparison

| Aspect | CatVTON | IDM-VTON |
|--------|---------|----------|
| **Inference Speed** | Faster (~20-30s) | Slower (~40-60s) |
| **Memory Usage** | Lower (8GB) | Higher (16GB) |
| **Detail Preservation** | Good for simple patterns | Better for complex patterns |
| **Setup Complexity** | Simple | Requires patches |

### What Works Well
- ✅ Both handle frontal poses reliably
- ✅ Solid color garments process well on both
- ✅ Clean backgrounds yield best results

### Challenges Identified
- ⚠️ **CatVTON:** Simplified complex patterns, faster but less detail
- ⚠️ **IDM-VTON:** Better detail but requires more memory optimization
- ⚠️ Both struggle with extreme poses (>45° rotation)
- ⚠️ Occlusion handling inconsistent (crossed arms, objects)

*Detailed visual comparisons coming after systematic evaluation.*

## 🛠️ Technical Notes

### Compatibility Fixes Applied (IDM-VTON)

The IDM-VTON notebook includes several patches for Kaggle compatibility:
```bash
# Diffusers import path fixes
sed -i 's/from diffusers.models.transformer_2d/from diffusers.models/g'

# Memory optimization
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

# Mask path corrections for VITON-HD dataset structure
# Image resizing to handle variable dimensions
# Garbage collection between batches
```

### Dataset Setup
Both notebooks use symlinks to avoid copying large datasets in Kaggle:
- VITON-HD: ~2GB (test set)
- DressCode: ~1.5GB (if used)

### Memory Management
IDM-VTON notebook includes aggressive cleanup:
```python
del images, prompt_embeds, image_embeds
torch.cuda.empty_cache()
gc.collect()
```

## 🔮 Next Steps

### Short-term (Current Focus)
- [x] Get both models running on Kaggle
- [x] Test on VITON-HD benchmark
- [x] Get it to work on in-the-wild images
- [ ] Curate in-the-wild dataset of 50-100 images
- [ ] Design VLM-based evaluation metric
- [ ] Document failure patterns systematically

### Medium-term (Planned)
- [ ] Refactor notebooks into modular Python scripts
- [ ] Build comparison dashboard

### Long-term (Research Direction)
- [ ] Systematic evaluation across pose categories
- [ ] Quantitative analysis of failure modes
- [ ] Metric validation with human judgments



## 📝 Related Work

### Models Explored
- **CatVTON** (2024): Zheng et al., "Concatenation Is All You Need for Virtual Try-On with Diffusion Models" - ICLR 2025
- **IDM-VTON** (2024): Choi et al., "Improving Diffusion Models for Authentic Virtual Try-on in the Wild"

### Future Models to Compare
- MV-VTON (AAAI 2025) - Multi-view specialization
- OOTDiffusion (2024) - Outfitting fusion approach

## 🤝 Contributing

This is a personal research exploration, but feedback welcome:
- Share interesting failure cases
- Suggest evaluation approaches
- Recommend test datasets

## 📧 Contact

For questions: shashank.devarmani@iiitb.ac.in

---

**Project Status:** 🔬 Active Implementation & Testing  
**Last Updated:** March 2026

**Note:** Notebooks are in exploratory/research state with implementation-specific patches. Refactoring into production-quality code is planned.
