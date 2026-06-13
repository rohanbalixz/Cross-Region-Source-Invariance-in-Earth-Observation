# Result receipts

One JSON file per experiment — the numbers reported in the study. `../../verify.py` re-derives the headline results from these without any data.

| Receipt | What it contains |
|---|---|
| `capacity_sweep.json` | Capacity x data-size sweep of the home-field gap (no under-fitting artefact). |
| `change_cnn_matrix.json` | CNN trained to predict change (not state) vs the parameter-free line. |
| `confound_n20_allpix.json` | Per-region difficulty vs change rate (Spearman, CI) across twenty regions. |
| `convlstm_disagreement.json` | Investigation of the Sprint D ConvLSTM rank-disagreement. Tests four hypotheses for why Co |
| `covariate_null.json` | Within-source vs cross-region prediction of per-tile FoM from physical covariates. |
| `fm_seg_finetuned.json` | Fine-tuned DINOv2 segmentation retention. |
| `fm_seg_matrix.json` | Frozen DINOv2 encoder segmentation retention. |
| `full_matrix_n20_cnn.json` | Twenty-region transfer matrix: source-invariance and the change-rate confound at scale. |
| `groupdro.json` | GroupDRO domain-generalisation baseline vs ERM pooling and single-source. |
| `imagery_loro_builtup_binary.json` | — |
| `imagery_multiseed.json` | Sentinel-2 built-up and 11-class segmentation retention (five seeds). |
| `input_representation_law.json` | Summary of retention by input family. |
| `landsat_temporal_matrix.json` | Landsat-history matched-input matrix (single seed). |
| `landsat_temporal_matrix_landsat_history.json` | — |
| `landsat_temporal_matrix_landsat_history_g128_multiseed.json` | Same, at finer 128-px resolution. |
| `landsat_temporal_matrix_landsat_history_g64_multiseed.json` | Raw multi-decade Landsat history input-swap retention. |
| `landsat_temporal_matrix_landsat_history_g64_n20_multiseed.json` | — |
| `loro_extent_cnn.json` | Leave-one-region-out pooling on the confound-free extent-IoU metric. |
| `lst_provenance_matrix.json` | Provenance dissociation on a climate target (MODIS land-surface temperature). |
| `mapping_stability.json` | Independent mapping-stability measure vs CNN cross-region retention. |
| `morph_baseline.json` | Parameter-free baselines (persistence / dilation / linear extrapolation) vs the trained CNN. |
| `multimetric_matrix.json` | Transfer matrices re-scored under extent IoU / F1 / change-F1. |
| `multiseed_gaps.json` | Per-seed home-field gaps. |
| `multiseed_matrix_cnn.json` | Five-seed CNN matrix: home-field gap, retention, source-invariance with error bars. |
| `multiseed_matrix_unet.json` | Five-seed U-Net matrix. |
| `multisource_loro_cnn.json` | — |
| `multitask_difficulty.json` | Source-invariance vs task difficulty across land-cover tasks. |
| `ndbi_builtup_matrix.json` | NDBI uniform-index input-swap retention. |
| `ndvi_builtup_g64_multiseed.json` | MODIS NDVI input-swap retention (degenerate control). |
| `ndvi_heldout_summary.json` | Held-out vegetation-dynamics nowcasting (field-level and change retention). |
| `ndvi_task_matrix.json` | NDVI nowcasting source-by-target matrix. |
| `new_region_validation.json` | Validation of the twelve newly-acquired regions. |
| `pangaea_reanalysis.json` | Re-analysis of the published PANGAEA benchmark (recomputed from its score table). |
| `pooled_resourced.json` | Pooled, resourced deflation control across 1x-8x model widths. |
| `prithvi_seg_matrix.json` | Frozen Prithvi-EO foundation-model segmentation retention. |
| `provenance_control.json` | Same-product NDBI nowcast: isolates provenance from autocorrelation. |
| `sar_matrix_g64_multiseed.json` | Sentinel-1 SAR input-swap retention (five seeds). |
| `seg_transfer_matrix.json` | Sentinel-2 -> WorldCover segmentation transfer matrix. |
| `spatial_holdout.json` | City-disjoint block holdout: within-city leakage does not manufacture the result. |
| `temporal_task_matrix.json` | Built-volume and population nowcasting matrices (temporal family). |
| `transfer_matrix_cnn.json` | CNN source-by-target transfer matrix (built-up nowcast), 8+CONUS regions. |
| `transfer_matrix_convlstm.json` | ConvLSTM source-by-target transfer matrix. |
| `transfer_matrix_unet.json` | U-Net source-by-target transfer matrix. |
| `worldcover_builtup_g64_multiseed.json` | WorldCover (all classes) -> GHSL built-up retention. |
| `worldcover_builtup_g64_nobuilt_multiseed.json` | WorldCover (built-up class withheld) -> GHSL built-up retention. |
