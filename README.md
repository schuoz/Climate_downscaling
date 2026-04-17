# Climate_downscaling
# Integrated Climate Downscaling

This repository provides an integrated pipeline for downscaling coarse atmospheric reanalysis data (like ERA5) to ultra-high resolutions (e.g. 100m). 

It bridges the deterministic topographic physics of the **CHELSA** algorithm with the non-linear generative capabilities of **Consistency Models** (OpenAI). By feeding CHELSA's high-resolution spatial covariates (e.g., windward/leeward exposure, precise elevation models, temperature lapse rates) as conditional *physics-guided* inputs into the Consistency Diffusion Model, we achieve 100m spatial fidelity without losing the complex variability modeled by Deep Learning.

## Architecture: Physics-Guided Machine Learning
1. **Data Processing:** Extracts coarse ERA5 data and generates/caches 100m topographic grids. Where possible, CHELSA grid computations use native Python arrays (`xarray`/`scipy`), falling back to `SAGA-GIS` APIs only where necessary.
2. **Machine Learning:** Modified UNet accepts coarse ERA5 sequences *plus* the generated static/dynamic covariate channels.
3. **Inference pipeline:** Uses tiling/chunking strategies to fit 100m grids on up to 26GB GPU instances.

## Installation
```bash
pip install -r requirements.txt
```
*Note: If some complex functions strictly require SAGA-GIS, ensure `saga_cmd` is installed and available in your pathway.*
