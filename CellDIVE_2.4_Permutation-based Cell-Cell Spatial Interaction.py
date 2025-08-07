#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Targeted cell types


# In[ ]:


# Tumour


# In[ ]:


import os
import numpy as np
import pandas as pd
import anndata as ad
import scimap as sm
import random
from itertools import combinations

# ─── REPRODUCIBILITY ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ─── USER SETTINGS ─────────────────────────────────────────────────────────────
COMBINED_CSV = (
    "/Volumes/My Passport/Spatial_Proteomics_data_final/"
    "cell_metadata_muspan_Tumour.csv"
)
PHENO      = "cell_type"
SAMPLE_COL = "slide_ID"

# just for plotting, not for filtering the data
TARGET_CELL_TYPES = [
   "FAP+ CAFs","aSMA+ CAFs","aSMA+FAP+ CAFs","Macrophages","Neutrophils",
    "exhausted CD4 T cells 2","CD8 T cells","exhausted CD8 T cells 1",
    "proliferating CD8 T cells","CD4 T cells","T-reg","exhausted CD4 T cells 1",
    "exhausted CD8 T cells 2","proliferating CD4 T cells","T cells","lymphatic CAFs",
    "Other Immune cells", "Endothelial cells", "Tumour cells", "proliferating Tumour cells"
]

region_name  = os.path.basename(COMBINED_CSV).rsplit(".", 1)[0]
OUT_DIR      = os.path.join(
    "/Volumes/My Passport/Spatial_Proteomics_data_final/"
    "Sci_Interaction_knn_targeted",
    region_name
)
OUT_PLOT_DIR = os.path.join(OUT_DIR, "slide_summary_heatmap")
OUT_CSV_DIR  = os.path.join(OUT_DIR, "slide_interactions_wide")
os.makedirs(OUT_PLOT_DIR, exist_ok=True)
os.makedirs(OUT_CSV_DIR, exist_ok=True)

# ─── PARAMETERS ────────────────────────────────────────────────────────────────
METHOD      = "knn"
knn         = 10
N_PERMS     = 1000
PVAL_THRESH = 0.05
LABEL       = f"slide_knn{knn}"

# ─── 1) LOAD & PREPARE ─────────────────────────────────────────────────────────
df = pd.read_csv(COMBINED_CSV)
df[SAMPLE_COL] = df[SAMPLE_COL].astype(str)
# assume CSV has x_um and y_um already

adata = ad.AnnData(
    X   = np.zeros((len(df),1)),
    obs = df.copy()
)

# ─── 2) RUN SPATIAL_INTERACTION ───────────────────────────────────────────────
adata = sm.tl.spatial_interaction(
    adata,
    x_coordinate    = "x_um",
    y_coordinate    = "y_um",
    phenotype       = PHENO,
    method          = METHOD,
    knn             = knn,
    permutation     = N_PERMS,
    imageid         = SAMPLE_COL,
    pval_method     = "zscore",
    label           = LABEL,
    verbose         = True
)

# ─── 3) PLOT SLIDE-LEVEL SUMMARY HEATMAP ───────────────────────────────────────
sm.pl.spatial_interaction(
    adata,
    spatial_interaction          = LABEL,
    p_val                        = PVAL_THRESH,
    summarize_plot               = True,
    return_data                  = False,
    saveDir                      = OUT_PLOT_DIR,
    fileName                     = f"{region_name}_slide_summary_heatmap.png",
    subset_phenotype             = TARGET_CELL_TYPES,
    subset_neighbour_phenotype   = TARGET_CELL_TYPES
)
print("Heatmap saved to:", OUT_PLOT_DIR)

# ─── 4) EXTRACT & SAVE FULL-DIRECTIONAL CSV ────────────────────────────────────
wide = adata.uns[LABEL].copy()

# identify score vs p-value columns
score_cols = [
    c for c in wide.columns
    if not c.startswith("pvalue_")
       and c not in ("phenotype","neighbour_phenotype")
]
pval_cols = [c for c in wide.columns if c.startswith("pvalue_")]

# melt to long form
df_score = wide.melt(
    id_vars    = ["phenotype","neighbour_phenotype"],
    value_vars = score_cols,
    var_name   = SAMPLE_COL,
    value_name = "interaction_score"
)
df_pval = wide.melt(
    id_vars    = ["phenotype","neighbour_phenotype"],
    value_vars = pval_cols,
    var_name   = "pval_col",
    value_name = "p_val"
)
df_pval[SAMPLE_COL] = df_pval["pval_col"].str.replace("pvalue_", "", regex=False)
df_pval.drop(columns=["pval_col"], inplace=True)

df_slide = (
    df_score
      .merge(df_pval, on=["phenotype","neighbour_phenotype", SAMPLE_COL])
      .rename(columns={
          "phenotype": "cluster_i",
          "neighbour_phenotype": "cluster_j"
      })
)
full_out = os.path.join(OUT_CSV_DIR, f"{region_name}_full_directional_interactions.csv")
df_slide.to_csv(full_out, index=False)
print("Full directional interactions saved to:", full_out)

# ─── 5) COLLAPSE A⟷B & B⟷A ────────────────────────────────────────────────────
collapsed = []
for slide, grp in df_slide.groupby(SAMPLE_COL):
    # build lookup dicts
    d_score = {(r.cluster_i, r.cluster_j): r.interaction_score for _,r in grp.iterrows()}
    d_pval  = {(r.cluster_i, r.cluster_j): r.p_val            for _,r in grp.iterrows()}
    for a, b in combinations(set(df[PHENO]), 2):
        v1 = d_score.get((a,b), np.nan)
        v2 = d_score.get((b,a), np.nan)
        p1 = d_pval.get((a,b), np.nan)
        p2 = d_pval.get((b,a), np.nan)
        scores = [v for v in (v1,v2) if not np.isnan(v)]
        mean_score = np.mean(scores) if scores else np.nan
        pvals = [p for p in (p1,p2) if not np.isnan(p)]
        combined_p = max(pvals) if pvals else np.nan
        collapsed.append({
            SAMPLE_COL:       slide,
            "pair":           f"{a}⟷{b}",
            "avg_score":      mean_score,
            "combined_p_val": combined_p
        })

collapsed_df = pd.DataFrame(collapsed)
collapsed_csv = os.path.join(OUT_CSV_DIR, f"{region_name}_collapsed_abba_interactions.csv")
collapsed_df.to_csv(collapsed_csv, index=False)
print("Collapsed A⟷B/B⟷A interactions saved to:", collapsed_csv)


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

TARGET_CELL_TYPES = [
   "FAP+ CAFs","aSMA+ CAFs","aSMA+FAP+ CAFs", "lymphatic CAFs", "Macrophages","Neutrophils", "CD8 T cells", 
    "proliferating CD8 T cells", "exhausted CD8 T cells 1", "exhausted CD8 T cells 2", "CD4 T cells", 
    "proliferating CD4 T cells", "exhausted CD4 T cells 1", "exhausted CD4 T cells 2", "T-reg", "T cells",
    "Other Immune cells", "Endothelial cells", "Tumour cells", "proliferating Tumour cells"
]

PVAL_THRESH = 0.05
DATA_PATH = '/Volumes/My Passport/Spatial_Proteomics_data_final/Sci_Interaction_knn_targeted/cell_metadata_muspan_Tumour/slide_interactions_wide/cell_metadata_muspan_Tumour_full_directional_interactions.csv'
OUT_PATH  = '/Volumes/My Passport/Spatial_Proteomics_data_final/Sci_Interaction_knn_targeted/cell_metadata_muspan_Tumour/slide_interactions_wide/Tumour_mean_interaction_heatmap_TARGET_CELLS_masked.png'

# Load data
df = pd.read_csv(DATA_PATH)

# Group by cell pair, aggregate across slides
agg = (
    df.groupby(['cluster_i', 'cluster_j'])
      .agg(mean_score=('interaction_score', 'mean'),
           max_pval=('p_val', 'max'))  # Use max pval as a conservative approach, only significant if all p < 0.05
      .reset_index()
)

# Pivot to square matrix for both values
score_mat = agg.pivot(index='cluster_i', columns='cluster_j', values='mean_score')
pval_mat  = agg.pivot(index='cluster_i', columns='cluster_j', values='max_pval')

# Subset and order for target cells only
score_mat = score_mat.loc[TARGET_CELL_TYPES, TARGET_CELL_TYPES]
pval_mat  = pval_mat.loc[TARGET_CELL_TYPES, TARGET_CELL_TYPES]

# Build a mask: True if NOT significant (p ≥ threshold)
nonsig_mask = (pval_mat >= PVAL_THRESH) | (pval_mat.isnull())

# Replace non-significant entries with np.nan so they'll be masked
masked_scores = score_mat.mask(nonsig_mask)

plt.figure(figsize=(14, 12), dpi=300)

# Create a custom diverging colormap: blue→white→red, white at 0
custom_cmap = LinearSegmentedColormap.from_list(
    'custom_coolwarm',
    [
        (0.00, '#2166ac'),  # blue at -1
        (0.50, '#ffffff'),  # white at 0
        (1.00, '#b2182b'),  # red at +1
    ]
)

# Plot heatmap with custom colormap
ax = sns.heatmap(
    masked_scores,
    cmap=custom_cmap,
    vmin=-1, vmax=1,
    square=True,
    cbar_kws={'label': 'Z-score'},
    mask=nonsig_mask,
    linewidths=0.5,
    linecolor='grey'
)

# Overlay dark grey squares for non-significant pairs
for i in range(len(TARGET_CELL_TYPES)):
    for j in range(len(TARGET_CELL_TYPES)):
        if nonsig_mask.iloc[i, j]:
            ax.add_patch(
                plt.Rectangle((j, i), 1, 1, color='#444444', lw=0)
            )

plt.title('Mean Cell–Cell Interaction Across Slides (Tumour): Significant Only')
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300)
plt.show()


# In[ ]:


# Peritumour


# In[ ]:


import os
import numpy as np
import pandas as pd
import anndata as ad
import scimap as sm
import random
from itertools import combinations

# ─── REPRODUCIBILITY ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ─── USER SETTINGS ─────────────────────────────────────────────────────────────
COMBINED_CSV = (
    "/Volumes/My Passport/Spatial_Proteomics_data_final/"
    "cell_metadata_muspan_Peritumour.csv"
)
PHENO      = "cell_type"
SAMPLE_COL = "slide_ID"

# just for plotting, not for filtering the data
TARGET_CELL_TYPES = [
   "FAP+ CAFs","aSMA+ CAFs","aSMA+FAP+ CAFs","Macrophages","Neutrophils",
    "exhausted CD4 T cells 2","CD8 T cells","exhausted CD8 T cells 1",
    "proliferating CD8 T cells","CD4 T cells","T-reg","exhausted CD4 T cells 1",
    "exhausted CD8 T cells 2","proliferating CD4 T cells","T cells","lymphatic CAFs",
    "Other Immune cells", "Endothelial cells", "Tumour cells", "proliferating Tumour cells"
]

region_name  = os.path.basename(COMBINED_CSV).rsplit(".", 1)[0]
OUT_DIR      = os.path.join(
    "/Volumes/My Passport/Spatial_Proteomics_data_final/"
    "Sci_Interaction_knn_targeted",
    region_name
)
OUT_PLOT_DIR = os.path.join(OUT_DIR, "slide_summary_heatmap")
OUT_CSV_DIR  = os.path.join(OUT_DIR, "slide_interactions_wide")
os.makedirs(OUT_PLOT_DIR, exist_ok=True)
os.makedirs(OUT_CSV_DIR, exist_ok=True)

# ─── PARAMETERS ────────────────────────────────────────────────────────────────
METHOD      = "knn"
knn         = 10
N_PERMS     = 1000
PVAL_THRESH = 0.05
LABEL       = f"slide_knn{knn}"

# ─── 1) LOAD & PREPARE ─────────────────────────────────────────────────────────
df = pd.read_csv(COMBINED_CSV)
df[SAMPLE_COL] = df[SAMPLE_COL].astype(str)
# assume CSV has x_um and y_um already

adata = ad.AnnData(
    X   = np.zeros((len(df),1)),
    obs = df.copy()
)

# ─── 2) RUN SPATIAL_INTERACTION ───────────────────────────────────────────────
adata = sm.tl.spatial_interaction(
    adata,
    x_coordinate    = "x_um",
    y_coordinate    = "y_um",
    phenotype       = PHENO,
    method          = METHOD,
    knn             = knn,
    permutation     = N_PERMS,
    imageid         = SAMPLE_COL,
    pval_method     = "zscore",
    label           = LABEL,
    verbose         = True
)

# ─── 3) PLOT SLIDE-LEVEL SUMMARY HEATMAP ───────────────────────────────────────
sm.pl.spatial_interaction(
    adata,
    spatial_interaction          = LABEL,
    p_val                        = PVAL_THRESH,
    summarize_plot               = True,
    return_data                  = False,
    saveDir                      = OUT_PLOT_DIR,
    fileName                     = f"{region_name}_slide_summary_heatmap.png",
    subset_phenotype             = TARGET_CELL_TYPES,
    subset_neighbour_phenotype   = TARGET_CELL_TYPES
)
print("Heatmap saved to:", OUT_PLOT_DIR)

# ─── 4) EXTRACT & SAVE FULL-DIRECTIONAL CSV ────────────────────────────────────
wide = adata.uns[LABEL].copy()

# identify score vs p-value columns
score_cols = [
    c for c in wide.columns
    if not c.startswith("pvalue_")
       and c not in ("phenotype","neighbour_phenotype")
]
pval_cols = [c for c in wide.columns if c.startswith("pvalue_")]

# melt to long form
df_score = wide.melt(
    id_vars    = ["phenotype","neighbour_phenotype"],
    value_vars = score_cols,
    var_name   = SAMPLE_COL,
    value_name = "interaction_score"
)
df_pval = wide.melt(
    id_vars    = ["phenotype","neighbour_phenotype"],
    value_vars = pval_cols,
    var_name   = "pval_col",
    value_name = "p_val"
)
df_pval[SAMPLE_COL] = df_pval["pval_col"].str.replace("pvalue_", "", regex=False)
df_pval.drop(columns=["pval_col"], inplace=True)

df_slide = (
    df_score
      .merge(df_pval, on=["phenotype","neighbour_phenotype", SAMPLE_COL])
      .rename(columns={
          "phenotype": "cluster_i",
          "neighbour_phenotype": "cluster_j"
      })
)
full_out = os.path.join(OUT_CSV_DIR, f"{region_name}_full_directional_interactions.csv")
df_slide.to_csv(full_out, index=False)
print("Full directional interactions saved to:", full_out)

# ─── 5) COLLAPSE A⟷B & B⟷A ────────────────────────────────────────────────────
collapsed = []
for slide, grp in df_slide.groupby(SAMPLE_COL):
    # build lookup dicts
    d_score = {(r.cluster_i, r.cluster_j): r.interaction_score for _,r in grp.iterrows()}
    d_pval  = {(r.cluster_i, r.cluster_j): r.p_val            for _,r in grp.iterrows()}
    for a, b in combinations(set(df[PHENO]), 2):
        v1 = d_score.get((a,b), np.nan)
        v2 = d_score.get((b,a), np.nan)
        p1 = d_pval.get((a,b), np.nan)
        p2 = d_pval.get((b,a), np.nan)
        scores = [v for v in (v1,v2) if not np.isnan(v)]
        mean_score = np.mean(scores) if scores else np.nan
        pvals = [p for p in (p1,p2) if not np.isnan(p)]
        combined_p = max(pvals) if pvals else np.nan
        collapsed.append({
            SAMPLE_COL:       slide,
            "pair":           f"{a}⟷{b}",
            "avg_score":      mean_score,
            "combined_p_val": combined_p
        })

collapsed_df = pd.DataFrame(collapsed)
collapsed_csv = os.path.join(OUT_CSV_DIR, f"{region_name}_collapsed_abba_interactions.csv")
collapsed_df.to_csv(collapsed_csv, index=False)
print("Collapsed A⟷B/B⟷A interactions saved to:", collapsed_csv)


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

TARGET_CELL_TYPES = [
   "FAP+ CAFs","aSMA+ CAFs","aSMA+FAP+ CAFs", "lymphatic CAFs", "Macrophages","Neutrophils", "CD8 T cells", 
    "proliferating CD8 T cells", "exhausted CD8 T cells 1", "exhausted CD8 T cells 2", "CD4 T cells", 
    "proliferating CD4 T cells", "exhausted CD4 T cells 1", "exhausted CD4 T cells 2", "T-reg", "T cells",
    "Other Immune cells", "Endothelial cells", "Tumour cells", "proliferating Tumour cells"
]

PVAL_THRESH = 0.05
DATA_PATH = '/Volumes/My Passport/Spatial_Proteomics_data_final/Sci_Interaction_knn_targeted/cell_metadata_muspan_Peritumour/slide_interactions_wide/cell_metadata_muspan_Peritumour_full_directional_interactions.csv'
OUT_PATH  = '/Volumes/My Passport/Spatial_Proteomics_data_final/Sci_Interaction_knn_targeted/cell_metadata_muspan_Peritumour/slide_interactions_wide/Peritumour_mean_interaction_heatmap_TARGET_CELLS_masked.png'

# Load data
df = pd.read_csv(DATA_PATH)

# Group by cell pair, aggregate across slides
agg = (
    df.groupby(['cluster_i', 'cluster_j'])
      .agg(mean_score=('interaction_score', 'mean'),
           max_pval=('p_val', 'max'))  # Use max pval as a conservative approach, only significant if all p < 0.05
      .reset_index()
)

# Pivot to square matrix for both values
score_mat = agg.pivot(index='cluster_i', columns='cluster_j', values='mean_score')
pval_mat  = agg.pivot(index='cluster_i', columns='cluster_j', values='max_pval')

# Subset and order for target cells only
score_mat = score_mat.loc[TARGET_CELL_TYPES, TARGET_CELL_TYPES]
pval_mat  = pval_mat.loc[TARGET_CELL_TYPES, TARGET_CELL_TYPES]

# Build a mask: True if NOT significant (p ≥ threshold)
nonsig_mask = (pval_mat >= PVAL_THRESH) | (pval_mat.isnull())

# Replace non-significant entries with np.nan so they'll be masked
masked_scores = score_mat.mask(nonsig_mask)

plt.figure(figsize=(14, 12), dpi=300)

# Create a custom diverging colormap: blue→white→red, white at 0
custom_cmap = LinearSegmentedColormap.from_list(
    'custom_coolwarm',
    [
        (0.00, '#2166ac'),  # blue at -1
        (0.50, '#ffffff'),  # white at 0
        (1.00, '#b2182b'),  # red at +1
    ]
)

# Plot heatmap with custom colormap
ax = sns.heatmap(
    masked_scores,
    cmap=custom_cmap,
    vmin=-1, vmax=1,
    square=True,
    cbar_kws={'label': 'Z-score'},
    mask=nonsig_mask,
    linewidths=0.5,
    linecolor='grey'
)

# Overlay dark grey squares for non-significant pairs
for i in range(len(TARGET_CELL_TYPES)):
    for j in range(len(TARGET_CELL_TYPES)):
        if nonsig_mask.iloc[i, j]:
            ax.add_patch(
                plt.Rectangle((j, i), 1, 1, color='#444444', lw=0)
            )
            

plt.title('Mean Cell–Cell Interaction Across Slides (Peritumour): Significant Only')
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300)
plt.show()


# In[ ]:


# Healthy


# In[ ]:


import os
import numpy as np
import pandas as pd
import anndata as ad
import scimap as sm
import random
from itertools import combinations

# ─── REPRODUCIBILITY ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ─── USER SETTINGS ─────────────────────────────────────────────────────────────
COMBINED_CSV = (
    "/Volumes/My Passport/Spatial_Proteomics_data_final/"
    "cell_metadata_muspan_Healthy.csv"
)
PHENO      = "cell_type"
SAMPLE_COL = "slide_ID"

# just for plotting, not for filtering the data
TARGET_CELL_TYPES = [
   "FAP+ CAFs","aSMA+ CAFs","aSMA+FAP+ CAFs","Macrophages","Neutrophils",
    "exhausted CD4 T cells 2","CD8 T cells","exhausted CD8 T cells 1",
    "proliferating CD8 T cells","CD4 T cells","T-reg","exhausted CD4 T cells 1",
    "exhausted CD8 T cells 2","proliferating CD4 T cells","T cells","lymphatic CAFs",
    "Other Immune cells", "Endothelial cells", "Tumour cells", "proliferating Tumour cells"
]

region_name  = os.path.basename(COMBINED_CSV).rsplit(".", 1)[0]
OUT_DIR      = os.path.join(
    "/Volumes/My Passport/Spatial_Proteomics_data_final/"
    "Sci_Interaction_knn_targeted",
    region_name
)
OUT_PLOT_DIR = os.path.join(OUT_DIR, "slide_summary_heatmap")
OUT_CSV_DIR  = os.path.join(OUT_DIR, "slide_interactions_wide")
os.makedirs(OUT_PLOT_DIR, exist_ok=True)
os.makedirs(OUT_CSV_DIR, exist_ok=True)

# ─── PARAMETERS ────────────────────────────────────────────────────────────────
METHOD      = "knn"
knn         = 10
N_PERMS     = 1000
PVAL_THRESH = 0.05
LABEL       = f"slide_knn{knn}"

# ─── 1) LOAD & PREPARE ─────────────────────────────────────────────────────────
df = pd.read_csv(COMBINED_CSV)
df[SAMPLE_COL] = df[SAMPLE_COL].astype(str)
# assume CSV has x_um and y_um already

adata = ad.AnnData(
    X   = np.zeros((len(df),1)),
    obs = df.copy()
)

# ─── 2) RUN SPATIAL_INTERACTION ───────────────────────────────────────────────
adata = sm.tl.spatial_interaction(
    adata,
    x_coordinate    = "x_um",
    y_coordinate    = "y_um",
    phenotype       = PHENO,
    method          = METHOD,
    knn             = knn,
    permutation     = N_PERMS,
    imageid         = SAMPLE_COL,
    pval_method     = "zscore",
    label           = LABEL,
    verbose         = True
)

# ─── 3) PLOT SLIDE-LEVEL SUMMARY HEATMAP ───────────────────────────────────────
sm.pl.spatial_interaction(
    adata,
    spatial_interaction          = LABEL,
    p_val                        = PVAL_THRESH,
    summarize_plot               = True,
    return_data                  = False,
    saveDir                      = OUT_PLOT_DIR,
    fileName                     = f"{region_name}_slide_summary_heatmap.png",
    subset_phenotype             = TARGET_CELL_TYPES,
    subset_neighbour_phenotype   = TARGET_CELL_TYPES
)
print("Heatmap saved to:", OUT_PLOT_DIR)

# ─── 4) EXTRACT & SAVE FULL-DIRECTIONAL CSV ────────────────────────────────────
wide = adata.uns[LABEL].copy()

# identify score vs p-value columns
score_cols = [
    c for c in wide.columns
    if not c.startswith("pvalue_")
       and c not in ("phenotype","neighbour_phenotype")
]
pval_cols = [c for c in wide.columns if c.startswith("pvalue_")]

# melt to long form
df_score = wide.melt(
    id_vars    = ["phenotype","neighbour_phenotype"],
    value_vars = score_cols,
    var_name   = SAMPLE_COL,
    value_name = "interaction_score"
)
df_pval = wide.melt(
    id_vars    = ["phenotype","neighbour_phenotype"],
    value_vars = pval_cols,
    var_name   = "pval_col",
    value_name = "p_val"
)
df_pval[SAMPLE_COL] = df_pval["pval_col"].str.replace("pvalue_", "", regex=False)
df_pval.drop(columns=["pval_col"], inplace=True)

df_slide = (
    df_score
      .merge(df_pval, on=["phenotype","neighbour_phenotype", SAMPLE_COL])
      .rename(columns={
          "phenotype": "cluster_i",
          "neighbour_phenotype": "cluster_j"
      })
)
full_out = os.path.join(OUT_CSV_DIR, f"{region_name}_full_directional_interactions.csv")
df_slide.to_csv(full_out, index=False)
print("Full directional interactions saved to:", full_out)

# ─── 5) COLLAPSE A⟷B & B⟷A ────────────────────────────────────────────────────
collapsed = []
for slide, grp in df_slide.groupby(SAMPLE_COL):
    # build lookup dicts
    d_score = {(r.cluster_i, r.cluster_j): r.interaction_score for _,r in grp.iterrows()}
    d_pval  = {(r.cluster_i, r.cluster_j): r.p_val            for _,r in grp.iterrows()}
    for a, b in combinations(set(df[PHENO]), 2):
        v1 = d_score.get((a,b), np.nan)
        v2 = d_score.get((b,a), np.nan)
        p1 = d_pval.get((a,b), np.nan)
        p2 = d_pval.get((b,a), np.nan)
        scores = [v for v in (v1,v2) if not np.isnan(v)]
        mean_score = np.mean(scores) if scores else np.nan
        pvals = [p for p in (p1,p2) if not np.isnan(p)]
        combined_p = max(pvals) if pvals else np.nan
        collapsed.append({
            SAMPLE_COL:       slide,
            "pair":           f"{a}⟷{b}",
            "avg_score":      mean_score,
            "combined_p_val": combined_p
        })

collapsed_df = pd.DataFrame(collapsed)
collapsed_csv = os.path.join(OUT_CSV_DIR, f"{region_name}_collapsed_abba_interactions.csv")
collapsed_df.to_csv(collapsed_csv, index=False)
print("Collapsed A⟷B/B⟷A interactions saved to:", collapsed_csv)


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

TARGET_CELL_TYPES = [
   "FAP+ CAFs","aSMA+ CAFs","aSMA+FAP+ CAFs", "lymphatic CAFs", "Macrophages","Neutrophils", "CD8 T cells", 
    "proliferating CD8 T cells", "exhausted CD8 T cells 1", "exhausted CD8 T cells 2", "CD4 T cells", 
    "proliferating CD4 T cells", "exhausted CD4 T cells 1", "exhausted CD4 T cells 2", "T-reg", "T cells",
    "Other Immune cells", "Endothelial cells", "Hepatocytes"
]

PVAL_THRESH = 0.05
DATA_PATH = '/Volumes/My Passport/Spatial_Proteomics_data_final/Sci_Interaction_knn_targeted/cell_metadata_muspan_Healthy/slide_interactions_wide/cell_metadata_muspan_Healthy_full_directional_interactions.csv'
OUT_PATH  = '/Volumes/My Passport/Spatial_Proteomics_data_final/Sci_Interaction_knn_targeted/cell_metadata_muspan_Healthy/slide_interactions_wide/Healthy_mean_interaction_heatmap_TARGET_CELLS_masked.png'

# Load data
df = pd.read_csv(DATA_PATH)

# Group by cell pair, aggregate across slides
agg = (
    df.groupby(['cluster_i', 'cluster_j'])
      .agg(mean_score=('interaction_score', 'mean'),
           max_pval=('p_val', 'max'))  # Use max pval as a conservative approach, only significant if all p < 0.05
      .reset_index()
)

# Pivot to square matrix for both values
score_mat = agg.pivot(index='cluster_i', columns='cluster_j', values='mean_score')
pval_mat  = agg.pivot(index='cluster_i', columns='cluster_j', values='max_pval')

# Subset and order for target cells only
score_mat = score_mat.loc[TARGET_CELL_TYPES, TARGET_CELL_TYPES]
pval_mat  = pval_mat.loc[TARGET_CELL_TYPES, TARGET_CELL_TYPES]

# Build a mask: True if NOT significant (p ≥ threshold)
nonsig_mask = (pval_mat >= PVAL_THRESH) | (pval_mat.isnull())

# Replace non-significant entries with np.nan so they'll be masked
masked_scores = score_mat.mask(nonsig_mask)

plt.figure(figsize=(14, 12), dpi=300)

# Create a custom diverging colormap: blue→white→red, white at 0
custom_cmap = LinearSegmentedColormap.from_list(
    'custom_coolwarm',
    [
        (0.00, '#2166ac'),  # blue at -1
        (0.50, '#ffffff'),  # white at 0
        (1.00, '#b2182b'),  # red at +1
    ]
)

# Plot heatmap with custom colormap
ax = sns.heatmap(
    masked_scores,
    cmap=custom_cmap,
    vmin=-1, vmax=1,
    square=True,
    cbar_kws={'label': 'Z-score'},
    mask=nonsig_mask,
    linewidths=0.5,
    linecolor='grey'
)

# Overlay dark grey squares for non-significant pairs
for i in range(len(TARGET_CELL_TYPES)):
    for j in range(len(TARGET_CELL_TYPES)):
        if nonsig_mask.iloc[i, j]:
            ax.add_patch(
                plt.Rectangle((j, i), 1, 1, color='#444444', lw=0)
            )
            

plt.title('Mean Cell–Cell Interaction Across Slides (Healthy): Significant Only')
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300)
plt.show()


# In[ ]:





# In[ ]:




