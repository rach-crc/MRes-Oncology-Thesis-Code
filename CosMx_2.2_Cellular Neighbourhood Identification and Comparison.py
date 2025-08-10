#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors, BallTree
import time
import sys

class Neighborhoods(object):
    def __init__(self, cells, ks, cluster_col, sum_cols, keep_cols, X='X:X', Y='Y:Y', reg='Exp', add_dummies=True, radius=None):
        np.random.seed(0)  # Set a random seed for reproducibility
        self.cells_nodumz = cells
        self.X = X
        self.Y = Y
        self.reg = reg
        self.keep_cols = keep_cols
        self.sum_cols = sum_cols
        self.ks = ks
        self.cluster_col = cluster_col
        self.n_neighbors = max(ks)
        self.exps = list(self.cells_nodumz[self.reg].unique())
        self.bool_add_dummies = add_dummies
        self.radius = radius  

    def add_dummies(self):
        c = self.cells_nodumz
        dumz = pd.get_dummies(c[self.cluster_col])
        keep = c[self.keep_cols + [self.reg]]
        self.cells = pd.concat([keep, dumz], axis=1)
        print("Columns after adding dummies:", self.cells.columns)

    def make_windows_knn(self, job):
        start_time, idx, tissue_name, indices = job
        job_start = time.time()

        print(f"Starting: {idx+1}/{len(self.exps)} : {self.exps[idx]}")

        tissue = self.tissue_group.get_group(tissue_name)
        to_fit = tissue.loc[indices][[self.X, self.Y]].values

        fit = NearestNeighbors(n_neighbors=self.n_neighbors + 1).fit(tissue[[self.X, self.Y]].values)
        m = fit.kneighbors(to_fit)
        distances, neighbors = m[0][:, 1:], m[1][:, 1:]  # Skip first neighbor (self)

        args = distances.argsort(axis=1)
        add = np.arange(neighbors.shape[0]) * neighbors.shape[1]
        sorted_indices = neighbors.flatten()[args + add[:, None]]
        neighbors_sorted = tissue.index.values[sorted_indices]  # Sorted neighbor indices

        index_mapping = {idx: i for i, idx in enumerate(self.cells.index)}
        neighbors_int = np.vectorize(index_mapping.get)(neighbors_sorted)

        end_time = time.time()
        print(f"Finishing: {idx+1}/{len(self.exps)} : {self.exps[idx]} {end_time - job_start} {end_time - start_time}")
        
        return neighbors_int

    def make_windows_radius(self, job):
        start_time, idx, tissue_name, indices = job
        job_start = time.time()

        print(f"Starting: {idx+1}/{len(self.exps)} : {self.exps[idx]}")

        tissue = self.tissue_group.get_group(tissue_name)
        to_fit = tissue.loc[indices][[self.X, self.Y]].values

        tree = BallTree(tissue[[self.X, self.Y]].values)
        neighbors = tree.query_radius(to_fit, r=self.radius)

        neighbors_int = []
        index_mapping = {idx: i for i, idx in enumerate(tissue.index)}
        for i, neigh in enumerate(neighbors):
            valid_neighbors = [index_mapping.get(tissue.index[neighbor]) for neighbor in neigh if neighbor != i]
            neighbors_int.append([n for n in valid_neighbors if n is not None])

        end_time = time.time()
        print(f"Finishing: {idx+1}/{len(self.exps)} : {self.exps[idx]} {end_time - job_start} {end_time - start_time}")

        return neighbors_int

    def get_tissue_chunks(self):
        self.tissue_group = self.cells[[self.X, self.Y, self.reg]].groupby(self.reg)
        tissue_chunks = [(time.time(), self.exps.index(t), t, a) for t, indices in self.tissue_group.groups.items() for a in np.array_split(indices, 1)]
        return tissue_chunks

    def k_windows(self, method='knn'):
        if self.bool_add_dummies:
            self.add_dummies()
        else:
            self.cells = self.cells_nodumz
        self.sum_cols = [col for col in self.sum_cols if col in self.cells.columns]
        sum_cols = list(self.sum_cols)
        for col in sum_cols:
            if col in self.keep_cols:
                self.cells[col + '_sum'] = self.cells[col]
                self.sum_cols.remove(col)
                self.sum_cols += [col + '_sum']

        values = self.cells[self.sum_cols].values.astype(np.float32)  # Use float32 for precision
        tissue_chunks = self.get_tissue_chunks()

        if method == 'knn':
            tissues = [self.make_windows_knn(job) for job in tissue_chunks]
        elif method == 'radius':
            tissues = [self.make_windows_radius(job) for job in tissue_chunks]
        else:
            raise ValueError("Method must be 'knn' or 'radius'")

        out_dict = {}
        for k in self.ks:
            for neighbors, job in zip(tissues, tissue_chunks):
                chunk = np.arange(len(neighbors))
                tissue_name = job[2]
                indices = job[3]
                
                if method == 'knn':
                    window = values[neighbors[chunk, :k].flatten()].reshape(len(chunk), k, len(self.sum_cols)).sum(axis=1)
                elif method == 'radius':
                    window = np.array([values[neigh].sum(axis=0) for neigh in neighbors])

                out_dict[(tissue_name, k)] = (window, indices)

        windows = {}
        for k in self.ks:
            window = pd.concat([pd.DataFrame(out_dict[(exp, k)][0], index=out_dict[(exp, k)][1], columns=self.sum_cols) for exp in self.exps], axis=0)
            window = window.loc[self.cells.index.values]
            window = pd.concat([self.cells[self.keep_cols], window], axis=1)
            windows[k] = window

        return windows


# In[ ]:


import os
import pandas as pd
import numpy as np
import scanpy as sc

# ─── CONFIG ────────────────────────────────────────────────────────────────────
ROOT     = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/Neighbourhoods"
CELLS_FN = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/cell_metadata_muspan_combined_rois_all.csv"

# 1) Load CSV
cells = pd.read_csv(CELLS_FN)

# Create new columns in µm
cells['x_um'] = cells['x_mm'] * 1000
cells['y_um'] = cells['y_mm'] * 1000

# 2) Decide what to keep
keep_cols = ['region', 'slide_ID', 'x_um', 'y_um']

# 3) One‑hot encode cell_type
cluster_col = 'cell_type'
dummy_cols  = pd.get_dummies(cells[cluster_col]).columns.tolist()

# 4) Instantiate Neighborhoods
nb = Neighborhoods(
    cells=cells,
    ks=[10],
    cluster_col=cluster_col,
    sum_cols=dummy_cols,
    keep_cols=keep_cols,
    X='x_um',
    Y='y_um',
    reg='ROI_ID',
    add_dummies=True,
    radius=None
)

# 5) Build the 10‑NN windows and merge back ROI_ID
windows_10nn = nb.k_windows(method='knn')
df_10nn = windows_10nn[10].merge(
    cells[['ROI_ID']],
    left_index=True, right_index=True, how='left'
)

# (Optional) Debug: inspect variances to catch zero‑variance features
print("Top 10 features by lowest variance:\n", df_10nn[dummy_cols].var().sort_values().head(10))

# 6) Build AnnData
X_nb     = df_10nn[dummy_cols].values.astype(np.float32)
adata_nb = sc.AnnData(X_nb)
adata_nb.obs["batch"] = df_10nn["slide_ID"].astype(str).values

# 7) One‑hot encode region (drop one level)
region_dummies = pd.get_dummies(df_10nn["region"], drop_first=True)
for col in region_dummies.columns:
    adata_nb.obs[col] = region_dummies[col].values.astype(np.float32)

# 8) ComBat on slide + region
sc.pp.combat(
    adata_nb,
    key="batch",
    covariates=region_dummies.columns.tolist()
)

# 9) Replace corrected values and save
df_10nn.loc[:, dummy_cols] = adata_nb.X
out_path = os.path.join(ROOT, "knn10_neighborhoods_batch_corrected.csv")
df_10nn.reset_index(drop=True)[
    ["region","slide_ID","ROI_ID"] + dummy_cols
].to_csv(out_path, index=False)

print(f"Saved batch‑corrected neighborhoods → {out_path}")
print(df_10nn.head())


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Paths
neigh_csv_path = '/Volumes/My Passport/Spatial_Transcriptomics_data_final/Neighbourhoods/knn10_neighborhoods_batch_corrected.csv'
cells_csv_path = '/Volumes/My Passport/Spatial_Transcriptomics_data_final/cell_metadata_muspan_combined_rois_all.csv'

# 1. Load both DataFrames
neigh_df = pd.read_csv(neigh_csv_path)
cells_df = pd.read_csv(cells_csv_path)

# Check that row counts match
if neigh_df.shape[0] != cells_df.shape[0]:
    raise RuntimeError("Row counts differ between neighborhood and cell CSVs.")

# Identify metadata columns vs. feature columns
metadata_cols = ['region', 'slide_ID', 'x_um', 'y_um', 'ROI_ID']
feature_cols = [c for c in neigh_df.columns if c not in metadata_cols]

# 2. Elbow plot: inertia for k=1..15
inertias = []
K_range = list(range(1, 16))
X = neigh_df[feature_cols].values

for k in K_range:
    km = KMeans(n_clusters=k, random_state=0, n_init='auto')
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, marker='o')
plt.xticks(K_range)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Plot for KMeans on Neighborhood Features')
plt.grid(True)
plt.show()


# In[ ]:


# 3. Choose k=10, fit KMeans, and assign clusters
optimal_k = 10
kmeans = KMeans(n_clusters=optimal_k, random_state=0, n_init='auto')
neigh_df['cluster'] = kmeans.fit_predict(X)

# Add the cluster labels back into the original cells DataFrame 
cells_df['cluster'] = neigh_df['cluster'].values

# Save outputs to /mnt/data
neigh_out = '/Volumes/My Passport/Spatial_Transcriptomics_data_final/Neighbourhoods/knn10_neighborhoods_with_clusters_10.csv'
cells_out = '/Volumes/My Passport/Spatial_Transcriptomics_data_final/cell_metadata_with_clusters_10_combat.csv'
neigh_df.to_csv(neigh_out, index=False)
cells_df.to_csv(cells_out, index=False)

print(f"Saved neighborhood data with clusters: {neigh_out}")
print(f"Saved original cells with clusters: {cells_out}")

# 4. Compute cluster means and apply z-score, Columns Ordered by Cluster Enrichment (Staircase)
# Compute cluster means (cluster x features matrix)
cluster_means = neigh_df.groupby('cluster')[feature_cols].mean()
# Z-score columns (cell type-wise) -- each column mean 0, std 1
cluster_means_z = (cluster_means - cluster_means.mean(axis=0)) / cluster_means.std(axis=0)

# For each feature (column), find the cluster with maximal z-score
max_cluster_per_col = cluster_means_z.values.argmax(axis=0)
# Order features so those enriched in cluster 0 are leftmost, then cluster 1, etc.
sorted_col_idx = np.lexsort((-cluster_means_z.values.max(axis=0), max_cluster_per_col))

sorted_feature_cols = [cluster_means_z.columns[i] for i in sorted_col_idx]
cluster_means_z_sorted = cluster_means_z[sorted_feature_cols]

plt.figure(figsize=(14, 6))
plt.imshow(cluster_means_z_sorted, aspect='auto', cmap='viridis')
plt.colorbar(label='Z-scored Average Neighbor Count')
plt.xticks(ticks=np.arange(len(sorted_feature_cols)), labels=sorted_feature_cols, rotation=90, fontsize=8)
plt.yticks(ticks=np.arange(cluster_means_z.shape[0]), labels=[f"Cluster {i}" for i in range(cluster_means_z.shape[0])], fontsize=10)
plt.title('Heatmap: Z‐scored Neighborhood Cell‐Type Counts per Cluster')
plt.tight_layout()
plt.savefig('/Volumes/My Passport/Spatial_Transcriptomics_data_final/Neighbourhoods/cluster_10_staircase_heatmap.png', dpi=250, bbox_inches='tight')
plt.show()


# In[ ]:


# Stacked bar plots for all regions based on cell_metadata_with_clusters_10_combat.csv


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load your clustered cell metadata
cells_csv = '/Volumes/My Passport/Spatial_Transcriptomics_data_final/cell_metadata_with_clusters_10_combat.csv'
df = pd.read_csv(cells_csv)

# 2. For each ROI (unique ROI_ID) in each region, compute the proportion of each cluster
# First, subset the columns we need
roi_groups = df.groupby(['region', 'ROI_ID'])

# Count cells per cluster within each ROI
roi_cluster_counts = roi_groups['cluster'].value_counts().unstack(fill_value=0)

# For each ROI, get the fraction of each cluster
roi_cluster_props = roi_cluster_counts.div(roi_cluster_counts.sum(axis=1), axis=0)  # Normalize per ROI

# 3. Average cluster proportions **per region**
region_props = roi_cluster_props.groupby('region').mean()

# 4. Save averaged cluster proportions for reproducibility
region_props.to_csv('/Volumes/My Passport/Spatial_Transcriptomics_data_final/Neighbourhoods/region_avg_cluster_proportions.csv')

# 5. Plot: stacked bar plot (each bar = region, segments = avg. cluster proportions)
plt.figure(figsize=(8, 7))
bottom = np.zeros(region_props.shape[0])

# Set color palette (optional: adjust for more/less clusters)
colors = plt.cm.tab10.colors if region_props.shape[1] <= 10 else plt.cm.tab20.colors

# Transpose for easier plotting (now: cluster as rows, region as columns)
region_props_T = region_props.T

for i, cluster in enumerate(region_props_T.index):
    plt.bar(region_props_T.columns, region_props_T.loc[cluster], 
            bottom=bottom, color=colors[i % len(colors)], label=f"Cluster {cluster}")
    bottom += region_props_T.loc[cluster]

plt.ylabel("Cluster proportion (mean across ROIs)")
plt.xlabel("Region")
plt.title("Average Cluster Composition per Region (Averaged Across ROIs)")
plt.xticks(rotation=20)
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.tight_layout()

# 6. Save the plot
plt.savefig('/Volumes/My Passport/Spatial_Transcriptomics_data_final/Neighbourhoods/region_avg_cluster_stackedbar.png', dpi=250, bbox_inches='tight')
plt.show()


# In[ ]:


# Generate {Region}_ROI_cluster_proportions.csv


# In[ ]:


import os
import pandas as pd

# ─── 1) INPUT/OUTPUT PATHS ────────────────────────────────────────────────────
input_csv = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/cell_metadata_with_clusters_10_combat.csv"
output_folder = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/Neighbourhoods"
os.makedirs(output_folder, exist_ok=True)

# ─── 2) LOAD THE CELL-LEVEL CSV ───────────────────────────────────────────────
df = pd.read_csv(input_csv)

# Ensure the key columns exist
assert set(["region","slide_ID","ROI_ID","cluster"]).issubset(df.columns), \
       "Your CSV must have columns: region, slide_ID, ROI_ID, cluster"

# ─── 3) GROUP-BY & RAW COUNTS ─────────────────────────────────────────────────
counts = (
    df
    .groupby(["region", "slide_ID", "ROI_ID", "cluster"])
    .size()
    .rename("cell_count")
    .reset_index()
)
pivot_counts = counts.pivot_table(
    index=["region", "slide_ID", "ROI_ID"],
    columns="cluster",
    values="cell_count",
    fill_value=0
)
pivot_counts.columns = [f"Cluster_{int(c)}" for c in pivot_counts.columns]
pivot_counts = pivot_counts.reset_index()

# ─── 4) CONVERT RAW COUNTS → PROPORTIONS ────────────────────────────────────────
cluster_cols = [c for c in pivot_counts.columns if c.startswith("Cluster_")]
pivot_counts["total_cells"] = pivot_counts[cluster_cols].sum(axis=1)
for c in cluster_cols:
    pivot_counts[c] = pivot_counts[c].astype(float) / pivot_counts["total_cells"]
pivot_counts = pivot_counts.drop(columns=["total_cells"])

# ─── 5) SPLIT BY REGION & WRITE CSVs ───────────────────────────────────────────
all_regions = pivot_counts["region"].unique().tolist()

for region_name in all_regions:
    df_region = pivot_counts[pivot_counts["region"] == region_name].copy()

    # ─────────── Make columns match screenshot 1 ───────────────────────
    df_region = df_region.rename(columns={
        "region": "Region",
        "slide_ID": "Slide",
        "ROI_ID":   "ROI"
    })

    def _ensure_prefix(val, prefix):
        s = str(val)
        if not s.startswith(prefix):
            return f"{prefix}{s}"
        return s

    df_region["Slide"] = df_region["Slide"].apply(lambda x: _ensure_prefix(x, "Slide_"))
    df_region["ROI"]   = df_region["ROI"].apply(lambda x: _ensure_prefix(x, "ROI_"))

    # Re‐order columns so the first three are Region, Slide, ROI
    cluster_cols = sorted([c for c in df_region.columns if c.startswith("Cluster_")])
    df_region = df_region[["Region", "Slide", "ROI"] + cluster_cols]
    # ──────────────────────────────────────────────────────────────────────

    # Save out:
    safe_region = region_name.replace(" ", "_")
    out_fname   = f"{safe_region}_ROI_cluster_proportions.csv"
    out_path    = os.path.join(output_folder, out_fname)
    df_region.to_csv(out_path, index=False)

    print(f"Saved → {out_path}")


# In[ ]:


import os
import glob
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# Define targeted clusters
TARGET_CLUSTERS = [0, 2, 7]
TARGET_COLS     = [f"Cluster_{i}" for i in TARGET_CLUSTERS]

# ─── 1) PATHS ────────────────────────────────────────────────────────────────
neigh_dir  = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/Neighbourhoods"
out_paired = os.path.join(neigh_dir, "paired_wilcoxon_fdr_target_clusters.csv")

# ─── 2) LOAD CSVs ─────────────────────────────────────────────────────────────
csv_paths = glob.glob(os.path.join(neigh_dir, "*_ROI_cluster_proportions.csv"))
df_all    = pd.concat((pd.read_csv(p) for p in csv_paths), ignore_index=True)

# ─── 3) MELT ONLY TARGETED CLUSTERS ──────────────────────────────────────────
df_long = df_all.melt(
    id_vars=["Region", "Slide", "ROI"],
    value_vars=TARGET_COLS,
    var_name="Cluster",
    value_name="frac"
)

# ─── 4) SLIDE-LEVEL MEANS ────────────────────────────────────────────────────
slide_means_long = (
    df_long
    .groupby(["Slide", "Region", "Cluster"], as_index=False)["frac"]
    .mean()
    .rename(columns={"frac": "mean_frac"})
)

# ─── 5) SET UP PAIRS ─────────────────────────────────────────────────────────
regions = sorted(slide_means_long["Region"].unique())
PAIRS   = list(combinations(regions, 2))
clusters = TARGET_COLS

# ─── 6) PAIRED WILCOXON + BH–FDR ─────────────────────────────────────────────
results = []
for r1, r2 in PAIRS:
    print(f"Testing: {r1} vs {r2}")

    df1 = slide_means_long[slide_means_long.Region == r1].pivot(
        index="Slide", columns="Cluster", values="mean_frac"
    )
    df2 = slide_means_long[slide_means_long.Region == r2].pivot(
        index="Slide", columns="Cluster", values="mean_frac"
    )
    common = df1.index.intersection(df2.index)

    # Count total ROIs for each region (not per-cluster)
    total_rois_r1 = df_all[df_all.Region == r1]["ROI"].nunique()
    total_rois_r2 = df_all[df_all.Region == r2]["ROI"].nunique()

    for cl in clusters:
        print(f"  {cl:15s} TOTAL ROIs – {r1}: {total_rois_r1}, {r2}: {total_rois_r2}")

        x = df1.loc[common, cl]
        y = df2.loc[common, cl]
        n = len(common)
        if n < 2:
            stat_w, p_w = np.nan, np.nan
        else:
            stat_w, p_w = wilcoxon(x, y)

        results.append({
            "Region1":     r1,
            "Region2":     r2,
            "Cluster":     cl,
            "n_slides":    n,
            "total_rois_r1":   total_rois_r1,
            "total_rois_r2":   total_rois_r2,
            "stat_wilcox": stat_w,
            "p_value":     p_w
        })
        print(f"       slides n={n:>2d} stat={stat_w:6.3f} p={p_w:.3g}\n")

# ─── 7) FDR correction ────────────────────────────────────────────────────────
df_res     = pd.DataFrame(results)
out_frames = []
for (r1, r2), grp in df_res.groupby(["Region1", "Region2"]):
    pvals = grp["p_value"].fillna(1.0).values
    reject, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    out_frames.append(grp.assign(p_adj_fdr=p_adj, reject_fdr=reject))

final = pd.concat(out_frames, ignore_index=True)
final.to_csv(out_paired, index=False)
print(f"All done! Paired Wilcoxon + FDR saved to: {out_paired}")


# In[ ]:


# Box plots
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─── CONFIG ────────────────────────────────────────────────────────────────
NEIGH_DIR   = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/Neighbourhoods"
FDR_CSV     = os.path.join(NEIGH_DIR, "paired_wilcoxon_fdr_target_clusters.csv")
OUTPUT_DIR  = os.path.join(NEIGH_DIR, "across_region_boxplots_clusters")
REGIONS     = ["Healthy", "Peritumour", "Tumour"]
PAIRS       = [
    ("Healthy", "Peritumour"),
    ("Healthy", "Tumour"),
    ("Peritumour", "Tumour"),
]
COLORS      = {"Healthy": "lightgreen", "Peritumour": "skyblue", "Tumour": "orange"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── TARGET CLUSTERS ────────────────────────────────────────────────────────
TARGET_CLUSTERS = [0, 2, 7]
TARGET_COLS     = [f"Cluster_{i}" for i in TARGET_CLUSTERS]

# ─── 1) LOAD FDR RESULTS ────────────────────────────────────────────────────
fdr = pd.read_csv(FDR_CSV)

# ─── 2) LOAD ROI‑LEVEL CSVs (no exclusion) ─────────────────────────────────
csv_paths = glob.glob(os.path.join(NEIGH_DIR, "*_ROI_cluster_proportions.csv"))
df_all    = pd.concat((pd.read_csv(p) for p in csv_paths), ignore_index=True)

# melt only the targeted clusters
df_long = df_all.melt(
    id_vars=["Region", "Slide", "ROI"],
    value_vars=TARGET_COLS,
    var_name="Cluster",
    value_name="frac"
)

# ─── 3) COMPUTE SLIDE‑LEVEL MEANS ───────────────────────────────────────────
slide_means_long = (
    df_long
    .groupby(["Slide", "Region", "Cluster"], as_index=False)["frac"]
    .mean()
    .rename(columns={"frac": "mean_frac"})
)

slide_means = (
    slide_means_long
    .pivot(index=["Slide", "Region"], columns="Cluster", values="mean_frac")
    .reset_index()
)

# only plot your targeted clusters
CLUSTERS = TARGET_COLS

# ─── 4) MAKE BOXPLOTS FOR EACH TARGETED CLUSTER ────────────────────────────
for cl in CLUSTERS:
    # collect per‐region arrays and corresponding Slide IDs
    data = []
    slide_ids_list = []
    for r in REGIONS:
        tmp = slide_means[slide_means.Region == r][["Slide", cl]].dropna()
        data.append(tmp[cl].values)
        slide_ids_list.append(tmp["Slide"].values)

    # compute whisker bounds for each region
    bounds = []
    for vals in data:
        if len(vals) > 0:
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            lb = q1 - 1.5 * iqr
            ub = q3 + 1.5 * iqr
        else:
            lb, ub = np.nan, np.nan
        bounds.append((lb, ub))

    fig, ax = plt.subplots(figsize=(6, 7))
    x = np.arange(len(REGIONS))

    # boxplot (without fliers)
    bp = ax.boxplot(
        data,
        positions=x,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(color='k', linewidth=1),
        whiskerprops=dict(color='k'),
        capprops=dict(color='k'),
        medianprops=dict(color='k')
    )
    for patch, r in zip(bp['boxes'], REGIONS):
        patch.set_facecolor(COLORS[r])
        patch.set_alpha(0.5)

    # overlay slide points
    for i, vals in enumerate(data):
        if len(vals):
            jitter = np.random.normal(0, 0.03, size=len(vals))
            ax.scatter(i + jitter, vals, color='k', s=9, alpha=0.7, zorder=3)

    # label outliers by Slide ID (only numeric suffix)
    for i, (vals, slide_ids, (lb, ub)) in enumerate(zip(data, slide_ids_list, bounds)):
        for val, sid in zip(vals, slide_ids):
            if val < lb or val > ub:
                sid_num = sid.split('_', 1)[1]  # drop "Slide_"
                ax.text(i, val, sid_num,
                        fontsize=6, ha='center', va='bottom')

    # axes & title
    ax.set_xticks(x)
    ax.set_xticklabels(REGIONS, rotation=45, ha='right')
    ax.set_ylabel("Mean fraction per slide")
    ax.set_title(cl)

    # annotate significance
    for (r1, r2) in PAIRS:
        sig = fdr[
            (fdr.Region1 == r1) &
            (fdr.Region2 == r2) &
            (fdr.Cluster == cl) &
            (fdr.reject_fdr == True)
        ]
        if not sig.empty:
            i1, i2 = REGIONS.index(r1), REGIONS.index(r2)
            y1 = data[i1].max() if len(data[i1]) else 0
            y2 = data[i2].max() if len(data[i2]) else 0
            top = max(y1, y2)
            h = (top - min(np.concatenate(data))) * 0.02 if np.concatenate(data).size else 0.1
            y0 = top + h
            y1b = top + 2*h
            ax.plot([i1, i1, i2, i2], [y0, y1b, y1b, y0], color='k', lw=1.5)
            ax.text((i1 + i2) / 2, y1b + 0.1*h,
                    "*", ha='center', va='bottom', fontsize=14)

    # save figure
    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, f"{cl}_box.png")
    fig.savefig(out_png, dpi=250)
    plt.close(fig)
    print(f"Saved boxplot for {cl} → {out_png}")


# In[ ]:


# Volcano Plots
import os
import glob
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ─── CONFIG ────────────────────────────────────────────────────────────────
NEIGH_DIR    = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/Neighbourhoods"
FDR_CSV      = os.path.join(NEIGH_DIR, "paired_wilcoxon_fdr_target_clusters.csv")
OUTDIR       = os.path.join(NEIGH_DIR, "volcano_plots_clusters")
REGION_PAIRS = [
    ("Healthy",   "Peritumour"),
    ("Healthy",   "Tumour"),
    ("Peritumour","Tumour"),
]
PALETTE = [
    "#e6194b","#3cb44b","#ffe119","#1E90FF","#911eb4","#bcf60c","#fac8e7","#e6beff","#f58231", "#70360f","#f04d63","#aaffc3" "#ffd8b1", "#130a7a","#808080","#000000","#75dbfa","#2f5aa1","#b3aa70",
    "#a64b7e","#40c2b9","#006d2c","#fdbf6f","#b37070","#4b0082",
    "#ffff99","#FF69B4","#d950ff","#e665b4","#89ccad","#d1309b"
]
os.makedirs(OUTDIR, exist_ok=True)

# ─── 1) LOAD FDR RESULTS ────────────────────────────────────────────────────
if not os.path.isfile(FDR_CSV):
    sys.exit(f"ERROR: cannot find FDR file:\n  {FDR_CSV}")
fdr = pd.read_csv(FDR_CSV)

# ensure we have the right columns
for col in ("Region1","Region2","Cluster","p_adj_fdr"):
    if col not in fdr.columns:
        sys.exit(f"ERROR: '{col}' missing from {FDR_CSV}")

# ─── 2) LOAD ROI‐LEVEL PROPORTIONS & COMPUTE SLIDE‐LEVEL MEANS ─────────────
# read in all per-region ROI CSVs
csv_paths = glob.glob(os.path.join(NEIGH_DIR, "*_ROI_cluster_proportions.csv"))
df_all = pd.concat((pd.read_csv(p) for p in csv_paths), ignore_index=True)

# melt to long form: one row per (Slide,Region,ROI,Cluster)
df_long = df_all.melt(
    id_vars=["Region","Slide","ROI"],
    value_vars=[c for c in df_all.columns if c.startswith("Cluster_")],
    var_name="Cluster",
    value_name="frac"
)

# compute slide‐level mean fraction per (Slide,Region,Cluster)
slide_means_long = (
    df_long
    .groupby(["Slide","Region","Cluster"], as_index=False)["frac"]
    .mean()
    .rename(columns={"frac":"mean_frac"})
)

# ─── 3) BUILD COLOR MAP ─────────────────────────────────────────────────────
cluster_order = fdr["Cluster"].unique().tolist()
color_map = {
    cl: PALETTE[i % len(PALETTE)]
    for i, cl in enumerate(cluster_order)
}

# ─── 4) PLOT VOLCANO FOR EACH REGION PAIR ───────────────────────────────────
for r1, r2 in REGION_PAIRS:
    print(f"Plotting volcano: {r1} vs {r2} …")
    sub = fdr[(fdr.Region1==r1) & (fdr.Region2==r2)].copy()
    if sub.empty:
        print(f"  → No tests for {r1} vs {r2}, skipping.")
        continue

    # 4a) compute median per cluster & region
    med = (
        slide_means_long
        .groupby(["Cluster","Region"])["mean_frac"]
        .median()
        .reset_index()
    )
    m1 = med[med.Region==r1].set_index("Cluster")["mean_frac"]
    m2 = med[med.Region==r2].set_index("Cluster")["mean_frac"]

    # 4b) effect size & p‐value transform
    sub["delta"]      = sub["Cluster"].map(m2) - sub["Cluster"].map(m1)
    sub["neg_log10p"] = -np.log10(sub["p_adj_fdr"].replace(0, 1e-300))

    # 5) make the scatter
    fig, ax = plt.subplots(figsize=(14,6))
    for cl in cluster_order:
        pts = sub[sub.Cluster==cl]
        ax.scatter(
            pts.delta, pts.neg_log10p,
            color=color_map[cl],
            edgecolors='black',
            linewidths=0.8,
            s=50, zorder=2
        )

    # FDR threshold line
    ax.axhline(-np.log10(0.05), ls='--', c='black', lw=1)

    # region arrow + labels
    lx, ux = ax.get_xlim(); ly, uy = ax.get_ylim()
    y0 = ly - 0.08*(uy-ly)
    ax.annotate(r1, xy=(lx, y0), ha='left',  va='center')
    ax.annotate(r2, xy=(ux, y0), ha='right', va='center')
    ax.annotate('', xy=(ux, y0), xytext=(lx, y0),
                arrowprops=dict(arrowstyle='<->', lw=1.5, color='black'))

    ax.set_xlabel(f"{r2} median – {r1} median")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title(f"Volcano: {r1} vs {r2}")

    # 6) legend on right, ordered by Δ ascending
    cluster_deltas = sub[['Cluster','delta']].drop_duplicates().sort_values('delta')
    legend_clusters = cluster_deltas['Cluster'].tolist()
    handles = [
        Line2D([0],[0],
               marker='o',
               color=color_map[ct],
               linestyle='',
               markeredgecolor='black',
               markersize=6)
        for ct in legend_clusters
    ]
    ax.legend(handles, legend_clusters,
              bbox_to_anchor=(1.02,1), loc='upper left', borderaxespad=0.)

    # 7) Save and close
    plt.tight_layout(rect=(0,0,0.8,1))
    out_png = os.path.join(
        OUTDIR,
        f"volcano_{r1.replace(' ','_')}_vs_{r2.replace(' ','_')}.png"
    )
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"  → saved {out_png}")


# In[ ]:





# In[ ]:


# All cluster UMAP and PCA.


# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

# for ComBat
import scanpy as sc

# ─── 0) CONFIGURATION ────────────────────────────────────────────────────────
# Folder that contains your three region‐specific CSVs:
#   └─ Peritumour_ROI_cluster_proportions.csv
#   └─ Tumour_ROI_cluster_proportions.csv
#   └─ Healthy_ROI_cluster_proportions.csv
ROOT    = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/Neighbourhoods"
# List the regions exactly as they appear in your filenames (and in the "Region" column).
REGIONS = ["Tumour", "Peritumour", "Healthy"]
# custom colors for each region
color_map = {
    "Tumour":     "orange",
    "Peritumour": "lightblue",
    "Healthy":    "lightgreen"
}
# Filename template. It will do: f"{region}_ROI_cluster_proportions.csv"
SUMMARY_FN = "{region}_ROI_cluster_proportions.csv"
# Where to dump all PCA/UMAP outputs
OUT_DIR = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/Neighbourhoods/UMAP_PCA_all_clusters_CLR_combat"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 1) READ & CONCATENATE ALL THREE REGION TABLES ────────────────────────────
dfs = []
for region in REGIONS:
    in_path = os.path.join(ROOT, SUMMARY_FN.format(region=region))
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Cannot find the file for region '{region}': {in_path}")
    df = pd.read_csv(in_path)
    dfs.append(df)
big_df = pd.concat(dfs, axis=0, ignore_index=True)

# ─── 2) SET INDEX & EXTRACT FEATURE MATRIX ──────────────────────────────────
meta_cols = ["Region", "Slide", "ROI"]
big_df    = big_df.fillna(0)
for c in meta_cols:
    if c not in big_df.columns:
        raise KeyError(f"Expected to see a column '{c}' in your CSV but it was missing.")
big_df  = big_df.set_index(meta_cols)
feature_cols = [c for c in big_df.columns if c.startswith("Cluster_")]
if not feature_cols:
    raise ValueError("No columns starting with 'Cluster_' were found.")
X = big_df[feature_cols].values
# ─── 2b) CLR → ComBat(batch & preserve region) → Scale ──────────────────────────────────────────────────────────────────────────────────
# CLR transform
eps    = 1e-6
X_safe = X + eps
logX   = np.log(X_safe)
X_clr  = logX - logX.mean(axis=1, keepdims=True)

# wrap in AnnData for ComBat
data_adt = sc.AnnData(X_clr)
data_adt.obs['batch']  = big_df.index.get_level_values('Slide')
data_adt.obs['region'] = big_df.index.get_level_values('Region')

# remove slide effects, preserve region differences
sc.pp.combat(data_adt, key='batch', covariates=['region'])

# corrected and standardized
X_corr = data_adt.X
Xs     = StandardScaler().fit_transform(X_corr)

# save combined table
combined_csv = os.path.join(OUT_DIR, "all_regions_ROI_cluster_proportions_combined.csv")
big_df.reset_index().to_csv(combined_csv, index=False)
print(f"Saved combined table (ROI × cluster proportions): {combined_csv}")

# ─── 3) PCA ─────────────────────────────────────────────────────────────────
pca    = PCA(n_components=2, random_state=0)
coords = pca.fit_transform(Xs)
loadings  = pd.DataFrame(pca.components_.T, index=feature_cols, columns=["PC1","PC2"])
explained = pd.Series(pca.explained_variance_ratio_, index=["PC1","PC2"], name="explained_variance_ratio")
coords_df = pd.DataFrame(coords, columns=["PC1","PC2"], index=big_df.index)
coords_df.to_csv(os.path.join(OUT_DIR, "all_regions_PCA_ROI_cluster_coords.csv"))
loadings.to_csv(os.path.join(OUT_DIR, "all_regions_PCA_ROI_cluster_loadings.csv"))
explained.to_csv(os.path.join(OUT_DIR, "all_regions_PCA_ROI_cluster_explained_variance.csv"))
print("⏺ PCA files saved")

# ─── 4) PLOT PCA (colored by Region) ─────────────────────────────────────────
pca_png = os.path.join(OUT_DIR, "all_regions_PCA_ROI_cluster_by_region.png")
plt.figure(figsize=(7, 6))
for region in REGIONS:
    mask = coords_df.index.get_level_values("Region") == region
    plt.scatter(coords[mask, 0], coords[mask, 1], label=region, s=12, alpha=0.7, color=color_map[region])
# enforce legend order
handles, labels = plt.gca().get_legend_handles_labels()
order = [labels.index(r) for r in REGIONS]
plt.legend([handles[i] for i in order], [labels[i] for i in order], fontsize="medium", frameon=False,
           bbox_to_anchor=(1.02,1), loc="upper left", title="Region")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("PCA of All Regions (ROI Cluster Proportions)")
plt.tight_layout(rect=[0.8, 0.8, 1, 1])
plt.savefig(pca_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"PCA scatter plot saved: {pca_png}")

# ─── 4b) PLOT PCA (colored by Slide) ─────────────────────────────────────────
slides = sorted(coords_df.index.get_level_values("Slide").unique())
cmap = plt.get_cmap("tab20")
slide_colors = dict(zip(slides, cmap(np.linspace(0,1,len(slides)))))
pca_slide_png = os.path.join(OUT_DIR, "all_regions_PCA_ROI_cluster_by_slide.png")
plt.figure(figsize=(7, 6))
for s in slides:
    mask = coords_df.index.get_level_values("Slide") == s
    plt.scatter(coords[mask,0], coords[mask,1], label=s, s=10, alpha=0.5, color=slide_colors[s])
plt.legend(bbox_to_anchor=(1.02,1), loc="upper left", frameon=False, title="Slide", fontsize="small")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("PCA of All Regions (ROI Cluster Proportions) by Slide")
plt.tight_layout(rect=[0.8, 0.8, 1, 1])
plt.savefig(pca_slide_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"PCA by Slide saved: {pca_slide_png}")

# ─── 5) UMAP ────────────────────────────────────────────────────────────────
reducer = umap.UMAP(n_neighbors=50, min_dist=0.5, spread=1, random_state=42)
Z = reducer.fit_transform(Xs)
umap_df = pd.DataFrame(Z, columns=["UMAP1","UMAP2"], index=big_df.index)
umap_df.to_csv(os.path.join(OUT_DIR, "all_regions_ROI_cluster_umap_coords.csv"))
print("UMAP coordinates saved")

# ─── 6) PLOT UMAP (colored by Region) ────────────────────────────────────────
umap_png = os.path.join(OUT_DIR, "all_regions_ROI_cluster_umap_by_region.png")
fig, ax = plt.subplots(figsize=(7, 6))
for region in REGIONS:
    mask = umap_df.index.get_level_values("Region") == region
    ax.scatter(Z[mask,0], Z[mask,1], s=12, alpha=0.7, label=region,
               color=color_map[region], edgecolors="none")
handles, labels = ax.get_legend_handles_labels()
order = [labels.index(r) for r in REGIONS]
ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize="medium",
          frameon=False, loc="upper left", bbox_to_anchor=(1.02,1), title="Region")
ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
ax.set_title("UMAP of All Regions (ROI Cluster Proportions)")
fig.tight_layout(rect=[0.8, 0.8, 1, 1])
fig.savefig(umap_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"UMAP scatter plot saved: {umap_png}")

# ─── 6b) PLOT UMAP (colored by Slide) ────────────────────────────────────────
umap_slide_png = os.path.join(OUT_DIR, "all_regions_ROI_cluster_umap_by_slide.png")
fig, ax = plt.subplots(figsize=(7, 6))
for s in slides:
    mask = umap_df.index.get_level_values("Slide") == s
    ax.scatter(Z[mask,0], Z[mask,1], label=s, s=10, alpha=0.5,
               color=slide_colors[s], edgecolors="none")
ax.legend(bbox_to_anchor=(1.02,1), loc="upper left", frameon=False,
          title="Slide", fontsize="small")
ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
ax.set_title("UMAP of All Regions (ROI Cluster Proportions) by Slide")
fig.tight_layout(rect=[0.8, 0.8, 1, 1])
fig.savefig(umap_slide_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"UMAP by Slide saved: {umap_slide_png}")


# In[ ]:


# Targeted cluster UMAP and PCA.


# In[ ]:


import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import numpy as np
import scanpy as sc

# ─── CONFIG ────────────────────────────────────────────────────────────────────
ROOT            = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/Neighbourhoods"
REGIONS         = ["Tumour", "Peritumour", "Healthy"]
SUMMARY_FN      = "{region}_ROI_cluster_proportions.csv"

TARGET_CLUSTERS = [0, 2, 7]
TARGET_COLS     = [f"Cluster_{i}" for i in TARGET_CLUSTERS]

color_map = {
    "Tumour":     "orange",
    "Peritumour": "lightblue",
    "Healthy":    "lightgreen"
}

OUT_BASE   = os.path.join(ROOT, "Targeted_PCA_UMAP_no_filter_RAW_combat")
os.makedirs(OUT_BASE, exist_ok=True)

# ─── 1) LOAD, FILTER, SAVE & CONCATENATE ──────────────────────────────────────
filtered_dfs = []
roi_counts_per_region = {}
for region in REGIONS:
    in_path = os.path.join(ROOT, SUMMARY_FN.format(region=region))
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Cannot find {in_path!r}")
    df = pd.read_csv(in_path).fillna(0)
    df["Region"] = region

    want = ["Region", "Slide", "ROI"] + TARGET_COLS
    missing = set(want) - set(df.columns)
    if missing:
        raise KeyError(f"{region} missing columns: {missing}")
    df_sub = df[want]

    # NO FILTER: keep all ROIs, even if all target clusters are 0
    df_sub = df_sub.reset_index(drop=True)

    # Save per‐region filtered (optional)
    out_reg = os.path.join(
        OUT_BASE,
        f"{region}_filtered_clusters_{'_'.join(map(str, TARGET_CLUSTERS))}.csv"
    )
    df_sub.to_csv(out_reg, index=False)
    print(f"Wrote filtered for {region} → {out_reg}")

    roi_counts_per_region[region] = df_sub.shape[0]
    filtered_dfs.append(df_sub)

print("\nROIs used from each region after filtering:")
for region in REGIONS:
    print(f"  {region}: {roi_counts_per_region[region]}")

# concatenate all regions
big_df      = pd.concat(filtered_dfs, ignore_index=True)
combined_csv = os.path.join(OUT_BASE, "all_regions_filtered_clusters.csv")
big_df.to_csv(combined_csv, index=False)
print(f"\nWrote combined filtered → {combined_csv}")

# re‐index for PCA/UMAP
big_df_idx = big_df.set_index(["Region", "Slide", "ROI"])
X  = big_df_idx[TARGET_COLS].values

# ─── 2) ComBat on raw proportions ─────────────────────────────────────────────
adata = sc.AnnData(X)
adata.obs['batch']  = big_df_idx.index.get_level_values('Slide')
adata.obs['region'] = big_df_idx.index.get_level_values('Region')
sc.pp.combat(adata, key='batch', covariates=['region'])
X_corr = adata.X

# ─── 2b) Standardize features (zero mean/unit variance) ───────────────────────
Xs = StandardScaler().fit_transform(X_corr)

# ─── 3) PCA ───────────────────────────────────────────────────────────────────
pca       = PCA(n_components=2, random_state=0)
coords    = pca.fit_transform(Xs)
loadings  = pd.DataFrame(pca.components_.T,
                         index=TARGET_COLS,
                         columns=["PC1", "PC2"])
explained = pd.Series(pca.explained_variance_ratio_,
                      index=["PC1", "PC2"],
                      name="explained_variance_ratio")

coords_df = pd.DataFrame(coords,
                         columns=["PC1", "PC2"],
                         index=big_df_idx.index)

coords_df.reset_index().to_csv(os.path.join(OUT_BASE, "PCA_coords.csv"), index=False)
loadings.to_csv(os.path.join(OUT_BASE, "PCA_loadings.csv"))
explained.to_csv(os.path.join(OUT_BASE, "PCA_explained_variance.csv"))
print("Saved PCA coords, loadings, explained‐variance")

# ─── 4) Plot PCA (by Region) ─────────────────────────────────────────────────
plt.figure(figsize=(7,6))
for region in REGIONS:
    mask = coords_df.index.get_level_values("Region") == region
    plt.scatter(
        coords[mask,0],
        coords[mask,1],
        label=region,
        s=12,
        alpha=0.7,
        color=color_map[region]
    )
plt.legend(
    bbox_to_anchor=(1.02,1),
    loc="upper left",
    frameon=False,
    title="Region"
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"PCA (clusters {TARGET_CLUSTERS})")
plt.tight_layout(rect=[0.8,0.8,1,1])
pca_png = os.path.join(OUT_BASE, "PCA_plot_by_region.png")
plt.savefig(pca_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"PCA plot (by region) → {pca_png}")

# ─── 4b) Plot PCA (by Slide) ────────────────────────────────────────────────
slides = coords_df.index.get_level_values("Slide").unique()
cmap   = plt.get_cmap("tab20")(np.linspace(0,1,len(slides)))
slide_color_map = dict(zip(slides, cmap))
plt.figure(figsize=(7,6))
for slide in slides:
    mask = coords_df.index.get_level_values("Slide") == slide
    plt.scatter(
        coords[mask,0],
        coords[mask,1],
        label=slide,
        s=10,
        alpha=0.5,
        color=slide_color_map[slide]
    )
plt.legend(
    bbox_to_anchor=(1.02,1),
    loc="upper left",
    frameon=False,
    title="Slide"
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"PCA (clusters {TARGET_CLUSTERS}, colored by Slide)")
plt.tight_layout(rect=[0.8,0.8,1,1])
pca_slide_png = os.path.join(OUT_BASE, "PCA_plot_by_slide.png")
plt.savefig(pca_slide_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"PCA plot (by Slide) → {pca_slide_png}")

# ─── 5) UMAP ──────────────────────────────────────────────────────────────────
reducer = umap.UMAP(n_neighbors=50, min_dist=0.5, spread=1, random_state=42)
Z       = reducer.fit_transform(Xs)
umap_df = pd.DataFrame(Z,
                       columns=["UMAP1", "UMAP2"],
                       index=big_df_idx.index)
umap_csv = os.path.join(OUT_BASE, "UMAP_coords.csv")
umap_df.reset_index().to_csv(umap_csv, index=False)
print(f"UMAP coords → {umap_csv}")

# ─── 6) Plot UMAP (by Region) ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7,6))
for region in REGIONS:
    mask = umap_df.index.get_level_values("Region") == region
    ax.scatter(
        Z[mask,0],
        Z[mask,1],
        s=12,
        alpha=0.7,
        label=region,
        color=color_map[region],
        edgecolors="none"
    )
ax.legend(
    bbox_to_anchor=(1.02,1),
    loc="upper left",
    frameon=False,
    title="Region"
)
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.set_title(f"UMAP (clusters {TARGET_CLUSTERS})")
plt.tight_layout(rect=[0.8,0.8,1,1])
umap_png = os.path.join(OUT_BASE, "UMAP_plot_by_region.png")
fig.savefig(umap_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"UMAP plot (by region) → {umap_png}")

# ─── 6b) Plot UMAP (by Slide) ────────────────────────────────────────────────
slides_umap = umap_df.index.get_level_values("Slide").unique()
plt.figure(figsize=(7,6))
for slide in slides_umap:
    mask = umap_df.index.get_level_values("Slide") == slide
    plt.scatter(
        Z[mask, 0],
        Z[mask, 1],
        label=slide,
        s=10,
        alpha=0.5,
        color=slide_color_map[slide]
    )
plt.legend(
    bbox_to_anchor=(1.02,1),
    loc="upper left",
    frameon=False,
    title="Slide"
)
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title(f"UMAP (clusters {TARGET_CLUSTERS}, colored by Slide)")
plt.tight_layout(rect=[0.8,0.8,1,1])
umap_slide_png = os.path.join(OUT_BASE, "UMAP_plot_by_slide.png")
plt.savefig(umap_slide_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"UMAP plot (by Slide) → {umap_slide_png}")


# In[ ]:




