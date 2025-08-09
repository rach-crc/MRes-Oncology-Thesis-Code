#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('writefile', 'New0607_trans_normalised_cell_counts_parallel.py', 'import os\nimport glob\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport muspan as ms\nfrom multiprocessing import Pool\n\n# ─── EDIT ONLY THIS ───────────────────────────────────────────────────\n# Change REGION to whichever folder you want (e.g. "Tumour", "Healthy", "Border", ...)\nREGION    = "Healthy"\nROOT_BASE = "/Volumes/My Passport/Spatial_Transcriptomics_data_final"\n# ──────────────────────────────────────────────────────────────────────\n\ndef process_roi_counts(csv_path, base_out, slide):\n    """\n    Reads one ROI CSV, computes normalized cell‐type fractions for original labels,\n    and writes out a single‐row CSV with columns:\n      Region, Slide, ROI, <raw_labels...>\n    Also saves a bar‐plot PNG in the same folder.\n    """\n    df = pd.read_csv(csv_path)\n\n    # 1) Extract ROI_ID (either from column or filename), then prefix "ROI_"\n    if \'ROI_ID\' in df.columns:\n        raw_id = str(df[\'ROI_ID\'].iloc[0])\n    else:\n        raw_id = os.path.splitext(os.path.basename(csv_path))[0]\n    roi_label = f"ROI_{raw_id}"\n\n    # 2) Build an ms.domain with (x_mm, y_mm) converted to μm\n    coords = df[[\'x_mm\', \'y_mm\']].to_numpy() * 1000.0  # mm → μm\n    domain = ms.domain(roi_label)\n    domain.add_points(coords, \'Cell centres\')\n\n    # 3) Add raw labels from "cell_type"\n    domain.add_labels(\'cell_type\', df[\'cell_type\'])\n\n    # 4) Compute normalized counts for raw labels\n    norm_raw, labels_raw = ms.summary_statistics.label_counts(\n        domain, label_name=\'cell_type\'\n    )\n    raw_counts = {lab: float(norm_raw[i]) for i, lab in enumerate(labels_raw)}\n\n    # 5) Combine metadata + counts into one ordered dict\n    meta = {\n        \'Region\': REGION,\n        \'Slide\':  slide,\n        \'ROI\':    roi_label\n    }\n    row = {**meta, **raw_counts}\n\n    # 6) Create a single‐row DataFrame and save to CSV\n    df_counts = pd.DataFrame([row])\n    out_dir = os.path.join(base_out, REGION, slide, roi_label)\n    os.makedirs(out_dir, exist_ok=True)\n\n    out_csv = os.path.join(out_dir, f"{roi_label}_normalized_counts.csv")\n    df_counts.to_csv(out_csv, index=False)\n\n    # 7) Create and save bar‐plot of raw labels only\n    fig, ax = plt.subplots(figsize=(10, 6))\n    keys = list(raw_counts.keys())\n    vals = [raw_counts[k] for k in keys]\n\n    ax.bar(range(len(vals)), vals)\n    ax.set_xticks(range(len(vals)))\n    ax.set_xticklabels(keys, rotation=90, fontsize=8)\n    ax.set_ylabel("Normalized count fraction")\n    ax.set_title(f"{REGION} / {slide} / {roi_label}")\n    fig.tight_layout()\n\n    out_png = os.path.join(out_dir, f"{roi_label}_normalized_counts.png")\n    fig.savefig(out_png)\n    plt.close(fig)\n\n    print(f"[{REGION} | {slide}] {roi_label} → saved {len(raw_counts)} labels (CSV + PNG)")\n\n\ndef run_slide(slide_path):\n    slide = os.path.basename(slide_path)\n    base_out = os.path.join(ROOT_BASE, "Normalized_cell_count")\n    csv_pattern = os.path.join(slide_path, "ROI_*.csv")\n    csvs = sorted(glob.glob(csv_pattern))\n    print(f"[{REGION} | {slide}] {len(csvs)} ROIs → computing normalized counts")\n\n    for path in csvs:\n        process_roi_counts(path, base_out, slide)\n\n    print(f"[{REGION} | {slide}] done")\n    return slide\n\n\nif __name__ == "__main__":\n    region_dir = os.path.join(ROOT_BASE, REGION)\n    if not os.path.isdir(region_dir):\n        raise FileNotFoundError(f"Region folder not found: {region_dir}")\n\n    slides = [\n        d for d in os.listdir(region_dir)\n        if d.startswith("Slide_") and os.path.isdir(os.path.join(region_dir, d))\n    ]\n    slide_paths = [os.path.join(region_dir, s) for s in sorted(slides)]\n\n    # Parallelize processing across slides (up to 4 at once)\n    with Pool(processes=4) as pool:\n        results = pool.map(run_slide, slide_paths)\n\n    print(f"All done for region \'{REGION}\':", results)\n')


# In[ ]:


# Done: Healthy, Tumour, Peritumour


# In[ ]:


# Per region cell type averaged proportions (3 times)
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ─── EDIT ONLY THIS ───────────────────────────────────────────────────
# Choose the region you want to average (e.g. "Tumour", "Healthy", "Border", etc.)
REGION    = "Healthy"
ROOT_BASE = "/Volumes/My Passport/Spatial_Transcriptomics_data_final"
# ──────────────────────────────────────────────────────────────────────

def find_normalized_csvs(input_dir):
    """
    Recursively find all *_normalized_counts.csv files under input_dir.
    """
    pattern = os.path.join(input_dir, "**", "*_normalized_counts.csv")
    return glob.glob(pattern, recursive=True)


def average_rois_in_directory(input_dir):
    """
    Read all *_normalized_counts.csv files under input_dir,
    drop metadata columns ['Region', 'Slide', 'ROI'],
    concatenate into a single DataFrame, fill missing label columns with zeros,
    and return the combined DataFrame (numeric columns only).
    """
    files = find_normalized_csvs(input_dir)
    if not files:
        raise ValueError(f"No normalized_counts CSV files found in {input_dir!r}")
    
    df_list = []
    for fp in files:
        df = pd.read_csv(fp)
        # Drop metadata columns before averaging:
        for meta_col in ("Region", "Slide", "ROI"):
            if meta_col in df.columns:
                df = df.drop(columns=meta_col)
        df_list.append(df)

    # Stack all ROI-level frames into one DataFrame
    combined = pd.concat(df_list, axis=0, sort=True)
    combined = combined.fillna(0)
    return combined


def compute_and_save_average(df, out_csv_path, out_plot_path, title):
    """
    Given a DataFrame df where each row is one ROI and each column is
    a normalized-count (float), compute the mean across all ROIs,
    save that one-row result to out_csv_path, and plot a bar plot to out_plot_path.
    Only original cell_type labels are plotted.
    """
    # 1) Compute mean across ROIs
    mean_series = df.mean(axis=0)
    mean_df = mean_series.to_frame(name="average_normalized_count")
    mean_df.to_csv(out_csv_path)

    # 2) Plot bars for all labels
    labels = mean_df.index.tolist()
    values = [mean_df.loc[lbl, "average_normalized_count"] for lbl in labels]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=250)
    ax.bar(range(len(values)), values)

    # Set x-axis ticks/labels
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)

    ax.set_ylabel("Average normalized count fraction")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_plot_path, dpi=300)
    plt.close(fig)


def main():
    # 1) Build path to "Normalized_cell_count/<REGION>"
    normalized_root = os.path.join(ROOT_BASE, "Normalized_cell_count", REGION)
    if not os.path.isdir(normalized_root):
        raise FileNotFoundError(f"Folder not found: {normalized_root!r}")

    # 2) Find & combine all per-ROI CSVs
    try:
        all_rois_df = average_rois_in_directory(normalized_root)
    except ValueError as e:
        print(f"[{REGION}] {e}")
        return

    # 3) Save the cross-ROI average CSV + bar-plot
    out_csv = os.path.join(normalized_root, f"{REGION}_average_normalized_counts.csv")
    out_png = os.path.join(normalized_root, f"{REGION}_average_normalized_counts.png")
    title   = f"{REGION} average normalized counts (all ROIs)"

    compute_and_save_average(all_rois_df, out_csv, out_png, title)
    print(f"[{REGION}] Saved cross-ROI averages for {len(all_rois_df)} ROIs → {out_csv!r}")

if __name__ == "__main__":
    main()


# In[ ]:


# [Tumour] Saved cross-ROI averages for 464 ROIs → '/Volumes/My Passport/Spatial_Transcriptomics_data_final/Normalized_cell_count/Tumour/Tumour_average_normalized_counts.csv'


# In[ ]:


# [Peritumour] Saved cross-ROI averages for 327 ROIs → '/Volumes/My Passport/Spatial_Transcriptomics_data_final/Normalized_cell_count/Peritumour/Peritumour_average_normalized_counts.csv'


# In[ ]:


# [Healthy] Saved cross-ROI averages for 442 ROIs → '/Volumes/My Passport/Spatial_Transcriptomics_data_final/Normalized_cell_count/Healthy/Healthy_average_normalized_counts.csv'


# In[ ]:


# Stacked bar plots for all regions based on three average_normalized_counts.csvs


# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── USER‐EDITABLE PARAMETERS ──────────────────────────────────────────────────
BASE_DIR = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/Normalized_cell_count"
REGIONS  = ["Healthy", "Peritumour", "Tumour"]
FNAME_TEMPLATE = "{region}_average_normalized_counts.csv"
OUT_CSV  = os.path.join(BASE_DIR, "avg_all_regions_all_celltypes_counts.csv")
OUT_PNG  = os.path.join(BASE_DIR, "avg_all_regions_original_celltypes_stacked.png")

# ─── NEW COLOR MAP (in desired plotting/legend order) ─────────────────────────
color_map = {
    "Central.venous.LSEC": "#b50404",
    "Periportal.LSEC": "#eb1c1c",
    "Endothelial.A": "#fc4444",
    "Endothelial.B": "#f57171",
    "Cholangiocyte": "#f57c16",
    "Epithelial.crypt": "#e39d62",
    "Epithelial.villi": "#f7cba6",
    "Glial": "#f5f50a",
    "Stroma.A": "#355704",
    "Stroma.B.2": "#518508",
    "Stroma.C": "#76ad28",
    "Stroma.D": "#a0db4d",
    "Hepatocyte.A": "#054a80",
    "Hepatocyte.B": "#1b5f94",
    "Hepatocyte.C": "#217dc4",
    "Hepatocyte.D": "#3fa0eb",
    "Hepatocyte.E": "#72bdf7",
    "Malignancy.epithelial.A": "#f25aba",
    "Malignancy.epithelial.B": "#fa96d5",
    "T.cell.CD8": "#540f61",
    "T.cell.CD4": "#81368f",
    "T.cell.regulatory": "#b384bd",
    "Plasma": "#a006bd",
    "B.cell": "#d928fa",
    "Plasmablast": "#e897f7",
    "NK.cell": "#e3cfe8",
    "Mast.cell": "#284a4d",
    "Conventional.dendritic.cell": "#40787d",
    "Plasmacytoid.dendritic.cell": "#51989e",
    "Macrophage": "#62b8bf",
    "Monocyte": "#9bcacf",
    "Neutrophil": "#92e8f0",
    "Pericyte": "#f5edc6",
    "Smooth.muscle": "#0cf0d3",
    "Stellate": "#e3e5e6"
}

# ─── LOAD AND COMBINE ───────────────────────────────────────────────────────────
region_series = {}
for region in REGIONS:
    path = os.path.join(BASE_DIR, region, FNAME_TEMPLATE.format(region=region))
    df = pd.read_csv(path, index_col=0)
    region_series[region] = df["average_normalized_count"]
combined = pd.concat(region_series, axis=1).fillna(0.0).T
combined.to_csv(OUT_CSV)
print(f"Combined CSV → {OUT_CSV!r}")

# ─── PLOT STACKED BAR, USING NEW COLOR MAP ────────────────────────────────────
# force plotting & legend order to match the color_map keys
orig_labels = list(color_map.keys())
regions     = combined.index.tolist()
x           = np.arange(len(regions))

fig, ax = plt.subplots(figsize=(10, 9), dpi=250)
bottom = np.zeros_like(x, dtype=float)

for lbl in orig_labels:
    heights = combined.get(lbl, pd.Series(0, index=regions)).values
    ax.bar(x, heights, bottom=bottom, color=color_map[lbl], width=0.6, label=lbl)
    bottom += heights

# formatting
ax.set_xticks(x)
ax.set_xticklabels(regions, rotation=20, fontsize=10)
ax.set_ylabel("Fraction (original cell types)")
ax.set_title("Stacked‐bar: Original cell‐type fractions")

# legend (already in correct order)
ax.legend(bbox_to_anchor=(1.02,1), loc='upper left', fontsize=8, frameon=False)

plt.tight_layout(rect=[0,0,0.85,1])
fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:


# Stats


# In[ ]:


# Combine ROI.csvs for each Region


# In[ ]:


get_ipython().run_cell_magic('writefile', 'Combine0607_region_level_counts.py', 'import os\nimport glob\nimport pandas as pd\n\n# ─── EDIT ONLY THIS ───────────────────────────────────────────────────\n# Base folder containing per‐ROI normalized counts:\nROOT_NORM = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/Normalized_cell_count"\n# ──────────────────────────────────────────────────────────────────────\n\ndef combine_region(region_dir):\n    """\n    For a given region folder, find all ROI_*_normalized_counts.csv under any Slide_* subfolder,\n    concatenate them into a single DataFrame, save that CSV in the region folder,\n    and print the number of columns excluding the three metadata columns.\n    """\n    region_name = os.path.basename(region_dir.rstrip("/"))\n    all_csvs = glob.glob(os.path.join(region_dir, "Slide_*", "*", "*_normalized_counts.csv"), recursive=False)\n\n    if not all_csvs:\n        print(f"[{region_name}] No ROI CSV files found under {region_dir!r}.")\n        return\n\n    df_list = []\n    for fp in sorted(all_csvs):\n        df = pd.read_csv(fp)\n        df_list.append(df)\n\n    # Concatenate all per‐ROI DataFrames (they share the same columns)\n    combined = pd.concat(df_list, axis=0, ignore_index=True, sort=False).fillna(0)\n\n    # Save combined CSV into the region folder\n    out_csv = os.path.join(region_dir, f"{region_name}_all_ROIs_normalized_counts.csv")\n    combined.to_csv(out_csv, index=False)\n\n    # Count columns excluding the three metadata columns: Region, Slide, ROI\n    total_cols = combined.shape[1]\n    metadata_cols = [c for c in ("Region", "Slide", "ROI") if c in combined.columns]\n    num_data_cols = total_cols - len(metadata_cols)\n\n    print(f"[{region_name}] Combined {len(df_list)} ROI files → saved \'{out_csv}\'.")\n    print(f"[{region_name}] Number of columns (excluding {metadata_cols}): {num_data_cols}\\n")\n\n\ndef main():\n    if not os.path.isdir(ROOT_NORM):\n        raise FileNotFoundError(f"Root folder not found: {ROOT_NORM!r}")\n\n    # List all immediate subdirectories (each should be a region)\n    regions = [\n        os.path.join(ROOT_NORM, d) for d in os.listdir(ROOT_NORM)\n        if os.path.isdir(os.path.join(ROOT_NORM, d))\n    ]\n    if not regions:\n        print(f"No region folders found under {ROOT_NORM!r}.")\n        return\n\n    for region_dir in sorted(regions):\n        combine_region(region_dir)\n\n\nif __name__ == "__main__":\n    main()\n')


# In[ ]:


get_ipython().system('pwd')


# In[ ]:


# [Healthy] Combined 442 ROI files → saved '/Volumes/My Passport/Spatial_Transcriptomics_data_final/Normalized_cell_count/Healthy/Healthy_all_ROIs_normalized_counts.csv'.
# [Healthy] Number of columns (excluding ['Region', 'Slide', 'ROI']): 34

# [Peritumour] Combined 327 ROI files → saved '/Volumes/My Passport/Spatial_Transcriptomics_data_final/Normalized_cell_count/Peritumour/Peritumour_all_ROIs_normalized_counts.csv'.
# [Peritumour] Number of columns (excluding ['Region', 'Slide', 'ROI']): 35

# [Tumour] Combined 464 ROI files → saved '/Volumes/My Passport/Spatial_Transcriptomics_data_final/Normalized_cell_count/Tumour/Tumour_all_ROIs_normalized_counts.csv'.
# [Tumour] Number of columns (excluding ['Region', 'Slide', 'ROI']): 33


# In[ ]:


# Plots for each slide/region: Per slide per region cell type porportions


# In[ ]:


import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ────────────── EDIT THIS PATH ──────────────
ROOT_DIR = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/Normalized_cell_count"
# ────────────────────────────────────────────

# ─── NEW COLOR MAP (in desired plotting/legend order) ─────────────────────────
color_map = {
    "Central.venous.LSEC":            "#b50404",
    "Periportal.LSEC":                "#eb1c1c",
    "Endothelial.A":                  "#fc4444",
    "Endothelial.B":                  "#f57171",
    "Cholangiocyte":                  "#f57c16",
    "Epithelial.crypt":               "#e39d62",
    "Epithelial.villi":               "#f7cba6",
    "Glial":                          "#f5f50a",
    "Stroma.A":                       "#355704",
    "Stroma.B.2":                     "#518508",
    "Stroma.C":                       "#76ad28",
    "Stroma.D":                       "#a0db4d",
    "Hepatocyte.A":                   "#054a80",
    "Hepatocyte.B":                   "#1b5f94",
    "Hepatocyte.C":                   "#217dc4",
    "Hepatocyte.D":                   "#3fa0eb",
    "Hepatocyte.E":                   "#72bdf7",
    "Malignancy.epithelial.A":        "#f25aba",
    "Malignancy.epithelial.B":        "#fa96d5",
    "T.cell.CD8":                     "#540f61",
    "T.cell.CD4":                     "#81368f",
    "T.cell.regulatory":              "#b384bd",
    "Plasma":                         "#a006bd",
    "B.cell":                         "#d928fa",
    "Plasmablast":                    "#e897f7",
    "NK.cell":                        "#e3cfe8",
    "Mast.cell":                      "#284a4d",
    "Conventional.dendritic.cell":    "#40787d",
    "Plasmacytoid.dendritic.cell":    "#51989e",
    "Macrophage":                     "#62b8bf",
    "Monocyte":                       "#9bcacf",
    "Neutrophil":                     "#92e8f0",
    "Pericyte":                       "#f5edc6",
    "Smooth.muscle":                  "#0cf0d3",
    "Stellate":                       "#e3e5e6"
}

# ────────────── Read and combine all region‐level CSVs ──────────────
pattern   = os.path.join(ROOT_DIR, "*", "*_all_ROIs_normalized_counts.csv")
all_files = sorted(glob.glob(pattern))
if not all_files:
    raise FileNotFoundError(f"No files found under {pattern}")

df_list = []
for fp in all_files:
    tmp = pd.read_csv(fp)
    for col in ["Region", "Slide", "ROI"]:
        if col not in tmp.columns:
            raise ValueError(f"File '{fp}' is missing column '{col}'")
    df_list.append(tmp)
df_all = pd.concat(df_list, ignore_index=True, sort=False).fillna(0)

# ────────────── Identify cell‐type columns ──────────────
metadata_cols  = ["Region", "Slide", "ROI"]
celltype_cols  = [c for c in df_all.columns if c not in metadata_cols]

# ensure color_map covers only those present, and preserve order
cols_to_plot = [ct for ct in color_map.keys() if ct in celltype_cols]
missing      = set(celltype_cols) - set(cols_to_plot)
if missing:
    raise ValueError(f"Missing color(s) for these cell types: {sorted(missing)}")

# ────────────── Aggregate by (Slide, Region) ──────────────
agg = (
    df_all
    .groupby(["Slide", "Region"])[celltype_cols]
    .mean()
    .reset_index()
)
agg.to_csv(os.path.join(ROOT_DIR, "slide_region_mean_cellproportions.csv"), index=False)
print("Saved slide_region_mean_cellproportions.csv.")

# ────────────── Prepare output directory ──────────────
output_dir = os.path.join(ROOT_DIR, "slide_region_barplots")
os.makedirs(output_dir, exist_ok=True)

# ────────────── Plot each slide ──────────────
for slide in sorted(agg["Slide"].unique()):
    df_slide = agg[agg["Slide"] == slide].set_index("Region").drop(columns="Slide")
    colors   = [color_map[c] for c in cols_to_plot]

    fig, ax = plt.subplots(figsize=(12, 10), dpi=200)
    bottom = pd.Series(0, index=df_slide.index)
    for ct, col in zip(cols_to_plot, colors):
        vals = df_slide[ct]
        ax.bar(df_slide.index, vals, bottom=bottom, color=col, width=0.75)
        bottom += vals

    ax.set_ylabel("Average Cell-Type Proportion\n(across all ROIs)", fontsize=12)
    ax.set_title(f"{slide} — Stacked Cell-Type Proportions by Region", fontsize=14)
    ax.set_xticklabels(df_slide.index, rotation=20, ha="center", fontsize=11)

    # legend in same order as stack
    patches = [
        Rectangle((0,0),1,1,facecolor=color_map[ct],edgecolor="none")
        for ct in cols_to_plot
    ]
    ax.legend(patches, cols_to_plot,
              bbox_to_anchor=(1.02,1), loc="upper left",
              fontsize=9, title="Cell-Type", title_fontsize=10,
              frameon=False)

    plt.tight_layout(rect=[0,0,0.80,1])
    out_png = os.path.join(output_dir, f"{slide}_region_stacked_bar.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f" • Saved slide plot → {out_png!r}")

# ────────────── Plot each region ──────────────
for region in sorted(agg["Region"].unique()):
    df_region = agg[agg["Region"] == region].set_index("Slide").drop(columns="Region")
    colors    = [color_map[c] for c in cols_to_plot]

    fig, ax = plt.subplots(figsize=(12, 10), dpi=250)
    bottom = pd.Series(0, index=df_region.index)
    for ct, col in zip(cols_to_plot, colors):
        vals = df_region[ct]
        ax.bar(df_region.index, vals, bottom=bottom, color=col, width=0.75)
        bottom += vals

    ax.set_ylabel("Average Cell-Type Proportion\n(across all ROIs)", fontsize=12)
    ax.set_title(f"{region} — Stacked Cell-Type Proportions by Slide", fontsize=14)
    ax.set_xticklabels(df_region.index, rotation=20, ha="center", fontsize=11)

    patches = [
        Rectangle((0,0),1,1,facecolor=color_map[ct],edgecolor="none")
        for ct in cols_to_plot
    ]
    ax.legend(patches, cols_to_plot,
              bbox_to_anchor=(1.02,1), loc="upper left",
              fontsize=9, title="Cell-Type", title_fontsize=10,
              frameon=False)

    plt.tight_layout(rect=[0,0,0.80,1])
    out_png = os.path.join(output_dir, f"{region}_slide_stacked_bar.png")
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f" • Saved region plot → {out_png!r}")

print(f"\n✅ All plots saved in: {output_dir}")


# In[ ]:


# 2.1.1 Normalise cell count per ROI based on all cells in a ROI


# In[ ]:


# Reads ROI‐level normalized‐counts for those 21 types.
# Include ROIs with any of the targeted cell type present.
# Computes slide‐level mean fractions based on ROIs left.
# Runs paired Wilcoxon tests per region‐pair.
# Applies BH‐FDR.
# Saves both slide_means_normalized_counts.csv and paired_tests_targeted_FDR.csv.
import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# ─── CONFIG ────────────────────────────────────────────────────────────────
ROOT_DIR    = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/Normalized_cell_count"
REGIONS     = ["Healthy", "Peritumour", "Tumour"]
PAIRS       = [
    ("Healthy", "Peritumour"),
    ("Healthy", "Tumour"),
    ("Peritumour", "Tumour"),
]
TARGET_CELL_TYPES = [
    "Stroma.B.2", "Macrophage", "Endothelial.B", "Plasma", "Plasmablast",
    "Conventional.dendritic.cell", "Periportal.LSEC",
    "Stroma.C", "Monocyte", "Stroma.D", "Endothelial.A",
    "Plasmacytoid.dendritic.cell", "Mast.cell", "B.cell",
    "T.cell.CD8", "T.cell.CD4", "Stroma.A", "T.cell.regulatory",
    "Neutrophil", "NK.cell","Central.venous.LSEC"
]
OUT_SLIDE   = os.path.join(ROOT_DIR, "slide_means_normalized_counts.csv")
OUT_PAIRED  = os.path.join(ROOT_DIR, "paired_tests_targeted_FDR.csv")
# ────────────────────────────────────────────────────────────────────────────────

# Track ROI counts per region
region_total_rois = {reg: 0 for reg in REGIONS}
region_used_rois  = {reg: 0 for reg in REGIONS}

# 1) Gather ROI-level fractions for ROIs with any target cell present
records = []
for region in REGIONS:
    pattern = os.path.join(ROOT_DIR, region, "Slide_*", "ROI_*", "*_normalized_counts.csv")
    for fp in glob.glob(pattern):
        region_total_rois[region] += 1
        df_roi = pd.read_csv(fp)
        slide = os.path.basename(os.path.dirname(os.path.dirname(fp)))
        roi   = os.path.basename(os.path.dirname(fp))
        # fetch or default zero
        vals = [float(df_roi[cell].iloc[0]) if cell in df_roi.columns else 0.0
                for cell in TARGET_CELL_TYPES]
        if not any(v > 0 for v in vals):
            continue
        region_used_rois[region] += 1
        for cell, val in zip(TARGET_CELL_TYPES, vals):
            records.append({
                "Region": region,
                "Slide":  slide,
                "ROI":    roi,
                "cell_type": cell,
                "frac":   val
            })

if not records:
    raise RuntimeError(f"No ROI CSVs with target cells found under {ROOT_DIR}")

# report ROI usage
print("ROI usage by region:")
for reg in REGIONS:
    tot = region_total_rois[reg]
    used = region_used_rois[reg]
    pct = f"{used/ tot:.1%}" if tot > 0 else "N/A"
    print(f"  {reg:10s}: used {used} of {tot} ROIs ({pct})")

# 2) Build DataFrame and compute slide-level means
df = pd.DataFrame(records)
slide_means_long = (
    df
    .groupby(["Slide","Region","cell_type"], as_index=False)["frac"]
    .mean()
)
# pivot to wide
slide_means = slide_means_long.pivot_table(
    index=["Slide","Region"],
    columns="cell_type",
    values="frac",
    fill_value=0
).reset_index()
slide_means.to_csv(OUT_SLIDE, index=False)
print(f"Saved slide-level means → {OUT_SLIDE}")

# 3) Paired Wilcoxon tests
results = []
for r1, r2 in PAIRS:
    print(f"\nTesting: {r1} vs {r2}")
    df1 = slide_means[slide_means.Region==r1].set_index("Slide")
    df2 = slide_means[slide_means.Region==r2].set_index("Slide")
    common = df1.index.intersection(df2.index)
    for cell in TARGET_CELL_TYPES:
        x = df1.loc[common, cell]
        y = df2.loc[common, cell]
        n = len(common)
        if n < 2:
            stat, p = np.nan, np.nan
        else:
            stat, p = wilcoxon(x, y)
        results.append({
            "Region1":   r1,
            "Region2":   r2,
            "cell_type": cell,
            "n":         n,
            "stat":      stat,
            "p_value":   p
        })
        print(f"  {cell:25s} n={n:>2d} stat={stat:6.3f} p={p:.3g}")

# 4) FDR correction and save
out_frames = []
for r1, r2 in PAIRS:
    sub = pd.DataFrame([r for r in results if r["Region1"]==r1 and r["Region2"]==r2])
    pvals = sub["p_value"].fillna(1.0).values
    reject, p_fdr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    sub["p_adj_fdr"]  = p_fdr
    sub["reject_fdr"] = reject
    out_frames.append(sub)

final = pd.concat(out_frames, ignore_index=True)
final.to_csv(OUT_PAIRED, index=False)
print(f"All done! Paired Wilcoxon + FDR saved to:\n  {OUT_PAIRED}")


# In[ ]:


# Raw p-adj annotation
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─── CONFIG ────────────────────────────────────────────────────────────────
BASE_DIR       = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/Normalized_cell_count"
SLIDE_MEANS    = os.path.join(BASE_DIR, "slide_means_normalized_counts.csv")
FDR_CSV        = os.path.join(BASE_DIR, "paired_tests_targeted_FDR.csv")
REGIONS        = ["Healthy","Peritumour","Tumour"]
PAIRS          = [
    ("Healthy","Peritumour"),
    ("Healthy","Tumour"),
    ("Peritumour","Tumour"),
]
TARGET_CELL_TYPES = [
    "Stroma.B.2", "Macrophage", "Endothelial.B", "Plasma", "Plasmablast",
    "Conventional.dendritic.cell", "Periportal.LSEC",
    "Stroma.C", "Monocyte", "Stroma.D", "Endothelial.A",
    "Plasmacytoid.dendritic.cell", "Mast.cell", "B.cell",
    "T.cell.CD8", "T.cell.CD4", "Stroma.A", "T.cell.regulatory",
    "Neutrophil", "NK.cell","Central.venous.LSEC"
]
OUTPUT_DIR     = os.path.join(BASE_DIR, "across_region_boxplots_targeted")
COLORS         = {"Healthy":"lightgreen","Peritumour":"skyblue","Tumour":"orange"}
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) Load slide‐level means & FDR results
df_means = pd.read_csv(SLIDE_MEANS)
fdr      = pd.read_csv(FDR_CSV)

# figure out which column holds slide IDs
if "slide_ID" in df_means.columns:
    slide_col = "slide_ID"
elif "Slide" in df_means.columns:
    slide_col = "Slide"
else:
    raise KeyError("Neither 'slide_ID' nor 'Slide' column found in slide_means")

# 2) For each cell‐type, make the box+overlay+value‐annotation plot
for cell in TARGET_CELL_TYPES:
    # build per-region dataframes, keeping slide IDs alongside the value
    region_dfs = []
    for r in REGIONS:
        df_r = df_means.loc[df_means.Region == r, [slide_col, cell]].dropna(subset=[cell]).copy()
        # normalize slide labels to just the numeric portion
        df_r[slide_col] = df_r[slide_col].astype(str).str.replace(r'^Slide_', '', regex=True)
        region_dfs.append(df_r)

    # extract numeric arrays for boxplot
    data = [ df_r[cell].values for df_r in region_dfs ]

    # ─── compute whisker bounds for each region ───────────────────────────────
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

    # 3) Boxplot without fliers
    fig, ax = plt.subplots(figsize=(6,7))
    x = np.arange(len(REGIONS))
    bp = ax.boxplot(
        data, positions=x, widths=0.6, patch_artist=True,
        showfliers=False,
        boxprops=dict(color='k',linewidth=1),
        whiskerprops=dict(color='k'),
        capprops=dict(color='k'),
        medianprops=dict(color='k')
    )
    for patch, r in zip(bp['boxes'], REGIONS):
        patch.set_facecolor(COLORS[r])
        patch.set_alpha(0.5)

    # 4) Overlay slide points AND annotate outliers
    for i, region_df in enumerate(region_dfs):
        arr       = region_df[cell].values
        slide_ids = region_df[slide_col].values
        lb, ub    = bounds[i]
        if len(arr):
            jitter = np.random.normal(0, 0.03, size=len(arr))
            for j, (val, sid) in enumerate(zip(arr, slide_ids)):
                xj = i + jitter[j]
                ax.scatter(xj, val, color='k', s=9, alpha=0.7, zorder=3)
                if not np.isnan(lb) and not np.isnan(ub) and (val < lb or val > ub):
                    ax.text(xj, val, sid,
                            fontsize=8, ha='center', va='bottom', clip_on=True)

    # 5) Axes and title
    ax.set_xticks(x)
    ax.set_xticklabels(REGIONS, rotation=45, ha='right')
    ax.set_ylabel("Mean fraction per slide")
    ax.set_title(cell)

    # 6) Add adjusted‐p labels (instead of stars)
    for (r1, r2) in PAIRS:
        row = fdr[
            (fdr.Region1 == r1) &
            (fdr.Region2 == r2) &
            (fdr.cell_type == cell)
        ]
        if not row.empty and row.reject_fdr.iloc[0]:
            # pull the adjusted p‐value and format it
            p_adj = row.p_adj_fdr.iloc[0]
            label = f"p-adj={p_adj:.3f}"
            i1, i2 = REGIONS.index(r1), REGIONS.index(r2)
            y1, y2 = (data[i1].max() if len(data[i1]) else 0,
                      data[i2].max() if len(data[i2]) else 0)
            top = max(y1, y2)
            h   = (top - min(np.concatenate(data))) * 0.01 if np.concatenate(data).size else 0.1
            y0  = top + 0.2*h
            y1b = top + 2*h
            # draw connector
            ax.plot([i1, i1, i2, i2], [y0, y1b, y1b, y0], color='k', lw=1)
            # annotate with numeric q‐value
            ax.text((i1+i2)/2, y1b + 0.05*h,
                    label, ha='center', va='bottom', fontsize=6)

    # 7) Save
    plt.tight_layout()
    out_png = os.path.join(
        OUTPUT_DIR,
        f"{cell.replace(' ','_')}_box.png"
    )
    fig.savefig(out_png, dpi=250)
    plt.close(fig)
    print(f"Saved boxplot for {cell} → {out_png}")


# In[ ]:


import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ─── CONFIG ────────────────────────────────────────────────────────────────
BASE_DIR        = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/Normalized_cell_count"
FDR_CSV         = os.path.join(BASE_DIR, "paired_tests_targeted_FDR.csv")
SLIDE_MEANS_CSV = os.path.join(BASE_DIR, "slide_means_normalized_counts.csv")
OUTDIR          = os.path.join(BASE_DIR, "volcano_plots_targeted_FDR")
os.makedirs(OUTDIR, exist_ok=True)

REGION_PAIRS = [
    ("Healthy", "Peritumour"),
    ("Healthy", "Tumour"),
    ("Peritumour", "Tumour")
]

PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#1E90FF", "#911eb4", "#bcf60c", "#fac8e7", "#e6beff", 
    "#f58231", "#70360f", "#f04d63", "#aaffc3", "#ffd8b1", "#130a7a", "#808080", "#000000", 
    "#75dbfa", "#2f5aa1", "#b3aa70", "#a64b7e", "#40c2b9", "#006d2c", "#fdbf6f", "#b37070", 
    "#4b0082", "#ffff99", "#FF69B4", "#d950ff", "#e665b4", "#89ccad", "#d1309b"
]
# ────────────────────────────────────────────────────────────────────────────

# 1) Load inputs
for path in (FDR_CSV, SLIDE_MEANS_CSV):
    if not os.path.isfile(path):
        sys.exit(f"ERROR: cannot find required file:\n  {path}")

df = pd.read_csv(FDR_CSV)
means = pd.read_csv(SLIDE_MEANS_CSV)

if 'cell_type' not in df.columns:
    sys.exit(f"ERROR: 'cell_type' missing from {FDR_CSV}")

# 2) Build color map
cell_order = df['cell_type'].unique().tolist()
color_map  = {ct: PALETTE[i % len(PALETTE)] for i, ct in enumerate(cell_order)}

# 3) Melt slide means into long form
long_means = means.melt(
    id_vars=["Slide", "Region"],
    value_vars=cell_order,
    var_name="cell_type",
    value_name="mean_fraction"
)

# 4) Volcano plot for each region pair
for r1, r2 in REGION_PAIRS:
    print(f"Plotting volcano: {r1} vs {r2} …")
    sub = df[(df.Region1 == r1) & (df.Region2 == r2)].copy()

    # 4a) compute medians per cell_type & region
    med = (
        long_means
        .groupby(["cell_type","Region"])["mean_fraction"]
        .median()
        .reset_index()
    )
    m1 = med[med.Region==r1].set_index("cell_type")["mean_fraction"]
    m2 = med[med.Region==r2].set_index("cell_type")["mean_fraction"]

    # 4b) effect size & FDR transform
    sub["delta"]      = sub["cell_type"].map(m2) - sub["cell_type"].map(m1)
    sub["neg_log10p"] = -np.log10(sub["p_adj_fdr"].replace(0, 1e-300))

    # 5) scatter
    fig, ax = plt.subplots(figsize=(14,6))
    for ct in cell_order:
        pts = sub[sub.cell_type==ct]
        ax.scatter(
            pts.delta, pts.neg_log10p,
            color=color_map[ct],
            edgecolors='black',
            linewidths=0.8,
            s=50,    # non-zero marker size
            zorder=2
        )

    # FDR threshold line
    ax.axhline(-np.log10(0.05), ls='--', c='black', lw=1)

    # region arrow + labels
    lx, ux = ax.get_xlim()
    ly, uy = ax.get_ylim()
    y0 = ly - 0.08*(uy-ly)
    ax.annotate(r1, xy=(lx, y0), ha='left',  va='center')
    ax.annotate(r2, xy=(ux, y0), ha='right', va='center')
    ax.annotate('', xy=(ux, y0), xytext=(lx, y0),
                arrowprops=dict(arrowstyle='<->', lw=1.5, color='black'))

    ax.set_xlabel(f"{r2} median – {r1} median")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title(f"Volcano: {r1} vs {r2}")

    # 6) split & sort by sign of Δ for two legends
    delta_series = (
        sub[['cell_type','delta']]
        .drop_duplicates()
        .set_index('cell_type')['delta']
    )

    neg_items = sorted(
        [(ct, d) for ct, d in delta_series.items() if d < 0],
        key=lambda x: x[1]
    )
    pos_items = sorted(
        [(ct, d) for ct, d in delta_series.items() if d >= 0],
        key=lambda x: x[1]
    )

    neg_cts = [ct for ct, _ in neg_items]
    pos_cts = [ct for ct, _ in pos_items]

    neg_handles = [
        Line2D([0],[0], marker='o', color=color_map[ct],
               linestyle='', markeredgecolor='black', markersize=6)
        for ct in neg_cts
    ]
    pos_handles = [
        Line2D([0],[0], marker='o', color=color_map[ct],
               linestyle='', markeredgecolor='black', markersize=6)
        for ct in pos_cts
    ]

    # 7) make room on right
    plt.subplots_adjust(right=0.60)

    # 8) left legend (Δ < 0)
    leg1 = ax.legend(
        neg_handles, neg_cts,
        title=r1,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.95),
        frameon=False
    )
    ax.add_artist(leg1)

    # 9) right legend (Δ ≥ 0)
    leg2 = ax.legend(
        pos_handles, pos_cts,
        title=r2,
        loc="upper left",
        bbox_to_anchor=(1.35, 0.95),
        frameon=False
    )

    # 10) save & close
    plt.tight_layout(rect=(0,0,0.9,1))
    out_file = os.path.join(
        OUTDIR,
        f"volcano_{r1.replace(' ','_')}_vs_{r2.replace(' ','_')}.png"
    )
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"  → saved {out_file}\n")


# In[ ]:


# 2.1.2 Normalised Child Cell Type Fractions within ROIs with High Parent Broad Cell Type Infiltration (Broad type niche)


# In[ ]:


get_ipython().run_cell_magic('writefile', 'Target_normalised_cell_counts_by_broadtype.py', 'import os\nimport glob\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom multiprocessing import Pool\n\n# ─── EDIT ONLY THIS ───────────────────────────────────────────────────\nREGION     = "Healthy"\nROOT_BASE  = "/Volumes/My Passport/Spatial_Transcriptomics_data_final"\n# name of the new output folder (instead of "Normalized_cell_count")\nOUT_FOLDER = "Normalized_by_broad_type"\n# the exact list of cell‐types you want fractions for:\nTARGET_CELL_TYPES = [\n    "Stroma.B.2", "Macrophage", "Endothelial.B", "Plasma", "Plasmablast",\n    "Conventional.dendritic.cell", "Periportal.LSEC",\n    "Stroma.C", "Monocyte", "Stroma.D", "Endothelial.A",\n    "Plasmacytoid.dendritic.cell", "Mast.cell", "B.cell",\n    "T.cell.CD8", "T.cell.CD4", "Stroma.A", "T.cell.regulatory",\n    "Neutrophil", "NK.cell","Central.venous.LSEC"\n]\n# ──────────────────────────────────────────────────────────────────────\n\ndef process_roi_counts(csv_path, slide):\n    """\n    For one ROI CSV:\n      - read in \'cell_type\' + \'broad_type\'\n      - compute for each TARGET_CELL_TYPES:\n          count(cell_type) / count(broad_type)\n      - write a 1×N CSV of those 17 fractions plus Region/Slide/ROI\n      - save a bar‐plot of them\n    """\n    df = pd.read_csv(csv_path)\n    # extract ROI label\n    raw_id   = str(df[\'ROI_ID\'].iloc[0])\n    roi_label = f"ROI_{raw_id}"\n\n    # compute raw counts\n    counts_type  = df[\'cell_type\'].value_counts()\n    counts_broad = df[\'broad_type\'].value_counts()\n\n    # build a map cell_type → its broad_type\n    type2broad = df[[\'cell_type\',\'broad_type\']] \\\n                   .drop_duplicates() \\\n                   .set_index(\'cell_type\')[\'broad_type\'] \\\n                   .to_dict()\n\n    # compute fractions\n    fracs = {}\n    for ct in TARGET_CELL_TYPES:\n        ct_count    = counts_type.get(ct, 0)\n        broad       = type2broad.get(ct)\n        broad_count = counts_broad.get(broad, 0)\n        fracs[ct] = ct_count / broad_count if broad_count > 0 else 0.0\n\n    # assemble output row\n    meta = {\n        \'Region\': REGION,\n        \'Slide\':  slide,\n        \'ROI\':    roi_label\n    }\n    row = {**meta, **fracs}\n    df_out = pd.DataFrame([row])\n\n    # prepare output folder\n    out_dir = os.path.join(ROOT_BASE, OUT_FOLDER, REGION, slide, roi_label)\n    os.makedirs(out_dir, exist_ok=True)\n\n    # save CSV\n    out_csv = os.path.join(out_dir, f"{roi_label}_by_broadtype.csv")\n    df_out.to_csv(out_csv, index=False)\n\n    # bar‐plot\n    fig, ax = plt.subplots(figsize=(10, 6))\n    labels = list(fracs.keys())\n    values = [fracs[k] for k in labels]\n    ax.bar(range(len(values)), values)\n    ax.set_xticks(range(len(values)))\n    ax.set_xticklabels(labels, rotation=90, fontsize=7)\n    ax.set_ylabel("Fraction within broad_type")\n    ax.set_title(f"{REGION} | {slide} | {roi_label}")\n    fig.tight_layout()\n    out_png = os.path.join(out_dir, f"{roi_label}_by_broadtype.png")\n    fig.savefig(out_png)\n    plt.close(fig)\n\n    print(f"[{REGION} | {slide}] {roi_label} → wrote {out_csv} + plot")\n\ndef run_slide(slide_path):\n    slide    = os.path.basename(slide_path)\n    pattern  = os.path.join(slide_path, "ROI_*.csv")\n    csvs     = sorted(glob.glob(pattern))\n    print(f"[{REGION} | {slide}] {len(csvs)} ROIs")\n    for p in csvs:\n        process_roi_counts(p, slide)\n    return slide\n\nif __name__ == "__main__":\n    region_dir = os.path.join(ROOT_BASE, REGION)\n    if not os.path.isdir(region_dir):\n        raise FileNotFoundError(f"Region not found: {region_dir}")\n\n    slides = [\n        d for d in os.listdir(region_dir)\n        if d.startswith("Slide_") and os.path.isdir(os.path.join(region_dir, d))\n    ]\n    slide_paths = [os.path.join(region_dir, s) for s in sorted(slides)]\n\n    with Pool(processes=4) as pool:\n        done = pool.map(run_slide, slide_paths)\n\n    print("Done for region", REGION, "slides:", done)\n')


# In[ ]:


# Done: Tumour, Peritumour, Healthy


# In[ ]:


# Reads ROI‐level normalized‐counts for those 21 types.
# Exclude ROIs based on Exclude_{region}_rois.csv.
# Computes slide‐level mean fractions based on ROIs left.
# Runs paired Wilcoxon tests per region‐pair.
# Applies BH‐FDR.
# Saves both slide_means_normalized_counts.csv and paired_tests_targeted_FDR.csv.
import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# ─── 1) CONFIG ───────────────────────────────────────────────────────────────
ROOT_DIR    = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/Normalized_by_broad_type"
EXCL_DIR    = os.path.abspath(os.path.join(ROOT_DIR, os.pardir))
REGIONS     = ["Healthy", "Peritumour", "Tumour"]
PAIRS       = [
    ("Healthy", "Peritumour"),
    ("Healthy", "Tumour"),
    ("Peritumour", "Tumour"),
]
TARGET_CELL_TYPES = [
    "Stroma.B.2", "Macrophage", "Plasma", "Plasmablast",
    "Conventional.dendritic.cell",
    "Stroma.C", "Monocyte", "Stroma.D",
    "Plasmacytoid.dendritic.cell", "Mast.cell", "B.cell",
    "T.cell.CD8", "T.cell.CD4", "Stroma.A", "T.cell.regulatory",
    "Neutrophil", "NK.cell"
]
BROAD_TYPES = ["Immune Cells", "CAFs"]
OUT_SLIDE   = os.path.join(ROOT_DIR, "slide_means_normalized_counts.csv")
OUT_PAIRED  = os.path.join(ROOT_DIR, "paired_tests_targeted_FDR.csv")
# ────────────────────────────────────────────────────────────────────────────────

# ─── 2) LOAD EXCLUSION SETS ────────────────────────────────────────────────────
exclude = {}
for reg in REGIONS:
    fn = os.path.join(EXCL_DIR, f"Exclude_{reg}_rois.csv")
    if os.path.exists(fn):
        df_ex = pd.read_csv(fn)
        # set of (broad_type, Slide, ROI)
        exclude[reg] = set(zip(df_ex["broad_type"], df_ex["Slide"], df_ex["ROI"]))
    else:
        exclude[reg] = set()

# Map each cell_type to its parent broad_type
broad_map = {}
for ct in ["Stroma.A", "Stroma.B.2", "Stroma.C", "Stroma.D"]:
    broad_map[ct] = "CAFs"
for ct in [
    "Macrophage", "Plasma", "Plasmablast",
    "Conventional.dendritic.cell", "Monocyte",
    "Plasmacytoid.dendritic.cell", "Mast.cell", "B.cell",
    "T.cell.CD8", "T.cell.CD4", "T.cell.regulatory",
    "Neutrophil", "NK.cell"
]:
    broad_map[ct] = "Immune Cells"
for ct in [
    "Endothelial.B", "Periportal.LSEC", "Endothelial.A", 
    "Central.venous.LSEC"
]:
    broad_map[ct] = "Endothelial"
# ────────────────────────────────────────────────────────────────────────────────

# ─── 3) READ ALL ROIs INTO ONE WIDE TABLE ──────────────────────────────────────
records = []
for region in REGIONS:
    pattern = os.path.join(ROOT_DIR, region, "Slide_*", "ROI_*", "*_by_broadtype.csv")
    for fp in glob.glob(pattern):
        df_roi = pd.read_csv(fp)
        slide = os.path.basename(os.path.dirname(os.path.dirname(fp)))  # e.g. "Slide_9"
        roi   = os.path.basename(os.path.dirname(fp))                   # e.g. "ROI_1492"
        row = {"Region": region, "Slide": slide, "ROI": roi}
        for col in TARGET_CELL_TYPES:
            row[col] = float(df_roi[col].iloc[0]) if col in df_roi.columns else 0.0
        records.append(row)

combined = pd.DataFrame(records)
if combined.empty:
    raise RuntimeError(f"No ROI files found under {ROOT_DIR}")

# ─── 4) MELT → FILTER OUT EXCLUDED ROIs → PIVOT BACK TO WIDE SLIDE MEANS ───────
df_long = combined.melt(
    id_vars=["Region","Slide","ROI"],
    value_vars=TARGET_CELL_TYPES,
    var_name="cell_type",
    value_name="frac"
)

def keep_row(r):
    bt = broad_map.get(r["cell_type"])
    if bt is None:
        return True
    return (bt, r["Slide"], r["ROI"]) not in exclude[r["Region"]]

mask = df_long.apply(keep_row, axis=1)
df_filtered = df_long[mask]

# ─── 4.1) EXPANDED ROI‐EXCLUSION SUMMARY ───────────────────────────────────────
print("\nROI exclusion summary (per broad_type):")

# build a header with your three BROAD_TYPES
print(f"{'Region':>10s}  {'Total':>7s}   "
      f"{'Immune Cells':>13s}   {'CAFs':>5s}   {'Endothelial':>11s}")

for reg in REGIONS:
    total = len(exclude[reg])
    n_imm = sum(1 for (bt,_,_) in exclude[reg] if bt == "Immune Cells")
    n_caf = sum(1 for (bt,_,_) in exclude[reg] if bt == "CAFs")
    n_end = sum(1 for (bt,_,_) in exclude[reg] if bt == "Endothelial")
    print(f"→ {reg:10s}  {total:7d}   "
          f"{n_imm:13d}   {n_caf:5d}   {n_end:11d}")
print()

# ─── 4.2) COMPUTE SLIDE‐LEVEL MEANS ───────────────────────────────────────────
slide_means_long = (
    df_filtered
    .groupby(["Slide","Region","cell_type"], as_index=False)["frac"]
    .mean()
)
slide_means = slide_means_long.pivot_table(
    index=["Slide","Region"],
    columns="cell_type",
    values="frac",
    fill_value=0
).reset_index()
slide_means.to_csv(OUT_SLIDE, index=False)
print(f"Saved slide-level means → {OUT_SLIDE}")

# ─── 5) PAIRED WILCOXON + BH‐FDR ────────────────────────────────────────────────
results = []
for r1, r2 in PAIRS:
    print(f"Testing: {r1} vs {r2}")
    df1 = slide_means[slide_means.Region==r1].set_index("Slide")
    df2 = slide_means[slide_means.Region==r2].set_index("Slide")
    common = df1.index.intersection(df2.index)
    for cell in TARGET_CELL_TYPES:
        x = df1.loc[common, cell]
        y = df2.loc[common, cell]
        n = len(common)
        if n < 2:
            stat, p = np.nan, np.nan
        else:
            stat, p = wilcoxon(x, y)
        results.append({
            "Region1":   r1,
            "Region2":   r2,
            "cell_type": cell,
            "n":         n,
            "stat":      stat,
            "p_value":   p
        })
        print(f"  {cell:25s} n={n:>2d} stat={stat:6.3f} p={p:.3g}")
    print()

# ─── 6) FDR CORRECTION & SAVE ─────────────────────────────────────────────────
out_frames = []
for r1, r2 in PAIRS:
    sub = pd.DataFrame([r for r in results if r["Region1"]==r1 and r["Region2"]==r2])
    pvals = sub["p_value"].fillna(1.0).values
    reject, p_fdr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    sub["p_adj_fdr"]  = p_fdr
    sub["reject_fdr"] = reject
    out_frames.append(sub)

final = pd.concat(out_frames, ignore_index=True)
final.to_csv(OUT_PAIRED, index=False)
print(f"All done! Paired Wilcoxon + FDR saved to:\n  {OUT_PAIRED}")



# In[ ]:


# p-adj version box plots
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─── CONFIG ────────────────────────────────────────────────────────────────
BASE_DIR       = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/Normalized_by_broad_type"
SLIDE_MEANS    = os.path.join(BASE_DIR, "slide_means_normalized_counts.csv")
FDR_CSV        = os.path.join(BASE_DIR, "paired_tests_targeted_FDR.csv")
REGIONS        = ["Healthy","Peritumour","Tumour"]
PAIRS          = [
    ("Healthy","Peritumour"),
    ("Healthy","Tumour"),
    ("Peritumour","Tumour"),
]
TARGET_CELL_TYPES = [
    "Stroma.B.2", "Macrophage", "Plasma", "Plasmablast",
    "Conventional.dendritic.cell",
    "Stroma.C", "Monocyte", "Stroma.D",
    "Plasmacytoid.dendritic.cell", "Mast.cell", "B.cell",
    "T.cell.CD8", "T.cell.CD4", "Stroma.A", "T.cell.regulatory",
    "Neutrophil", "NK.cell"
]
OUTPUT_DIR     = os.path.join(BASE_DIR, "across_region_boxplots_targeted")
COLORS         = {"Healthy":"lightgreen","Peritumour":"skyblue","Tumour":"orange"}
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) Load slide‐level means & FDR results
df_means = pd.read_csv(SLIDE_MEANS)
fdr      = pd.read_csv(FDR_CSV)

# figure out which column holds slide IDs
if "slide_ID" in df_means.columns:
    slide_col = "slide_ID"
elif "Slide" in df_means.columns:
    slide_col = "Slide"
else:
    raise KeyError("Neither 'slide_ID' nor 'Slide' column found in slide_means")

# 2) For each cell‐type, make the box+overlay+value‐annotation plot
for cell in TARGET_CELL_TYPES:
    # build per-region dataframes, keeping slide IDs alongside the value
    region_dfs = []
    for r in REGIONS:
        df_r = df_means.loc[df_means.Region == r, [slide_col, cell]].dropna(subset=[cell]).copy()
        # normalize slide labels to just the numeric portion
        df_r[slide_col] = df_r[slide_col].astype(str).str.replace(r'^Slide_', '', regex=True)
        region_dfs.append(df_r)

    # extract numeric arrays for boxplot
    data = [ df_r[cell].values for df_r in region_dfs ]

    # ─── compute whisker bounds for each region ───────────────────────────────
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

    # 3) Boxplot without fliers
    fig, ax = plt.subplots(figsize=(6,7))
    x = np.arange(len(REGIONS))
    bp = ax.boxplot(
        data, positions=x, widths=0.6, patch_artist=True,
        showfliers=False,
        boxprops=dict(color='k',linewidth=1),
        whiskerprops=dict(color='k'),
        capprops=dict(color='k'),
        medianprops=dict(color='k')
    )
    for patch, r in zip(bp['boxes'], REGIONS):
        patch.set_facecolor(COLORS[r])
        patch.set_alpha(0.5)

    # 4) Overlay slide points AND annotate outliers
    for i, region_df in enumerate(region_dfs):
        arr       = region_df[cell].values
        slide_ids = region_df[slide_col].values
        lb, ub    = bounds[i]
        if len(arr):
            jitter = np.random.normal(0, 0.03, size=len(arr))
            for j, (val, sid) in enumerate(zip(arr, slide_ids)):
                xj = i + jitter[j]
                ax.scatter(xj, val, color='k', s=9, alpha=0.7, zorder=3)
                if not np.isnan(lb) and not np.isnan(ub) and (val < lb or val > ub):
                    ax.text(xj, val, sid,
                            fontsize=6, ha='center', va='bottom', clip_on=True)

    # 5) Axes and title
    ax.set_xticks(x)
    ax.set_xticklabels(REGIONS, rotation=45, ha='right')
    ax.set_ylabel("Mean fraction per slide")
    ax.set_title(cell)

    # 6) Add adjusted‐p labels (instead of stars)
    for (r1, r2) in PAIRS:
        row = fdr[
            (fdr.Region1 == r1) &
            (fdr.Region2 == r2) &
            (fdr.cell_type == cell)
        ]
        if not row.empty and row.reject_fdr.iloc[0]:
            # pull the adjusted p‐value and format it
            p_adj = row.p_adj_fdr.iloc[0]
            label = f"p-adj={p_adj:.3f}"
            i1, i2 = REGIONS.index(r1), REGIONS.index(r2)
            y1, y2 = (data[i1].max() if len(data[i1]) else 0,
                      data[i2].max() if len(data[i2]) else 0)
            top = max(y1, y2)
            h   = (top - min(np.concatenate(data))) * 0.01 if np.concatenate(data).size else 0.1
            y0  = top + 0.2*h
            y1b = top + 2*h
            # draw connector
            ax.plot([i1, i1, i2, i2], [y0, y1b, y1b, y0], color='k', lw=1)
            # annotate with numeric q‐value
            ax.text((i1+i2)/2, y1b + 0.05*h,
                    label, ha='center', va='bottom', fontsize=7)

    # 7) Save
    plt.tight_layout()
    out_png = os.path.join(
        OUTPUT_DIR,
        f"{cell.replace(' ','_')}_box.png"
    )
    fig.savefig(out_png, dpi=250)
    plt.close(fig)
    print(f"Saved boxplot for {cell} → {out_png}")


# In[ ]:


# Volcano Plots
import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ─── CONFIG ────────────────────────────────────────────────────────────────
BASE_DIR        = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/Normalized_by_broad_type"
FDR_CSV         = os.path.join(BASE_DIR, "paired_tests_targeted_FDR.csv")
SLIDE_MEANS_CSV = os.path.join(BASE_DIR, "slide_means_normalized_counts.csv")
OUTDIR          = os.path.join(BASE_DIR, "volcano_plots_targeted_FDR")
os.makedirs(OUTDIR, exist_ok=True)

REGION_PAIRS = [
    ("Healthy", "Peritumour"),
    ("Healthy", "Tumour"),
    ("Peritumour", "Tumour")
]

PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#1E90FF", "#911eb4", "#bcf60c", "#fac8e7", "#e6beff",
    "#f58231", "#70360f", "#f04d63", "#aaffc3", "#ffd8b1", "#130a7a", "#808080", "#000000",
    "#75dbfa", "#2f5aa1", "#b3aa70", "#a64b7e", "#40c2b9", "#006d2c", "#fdbf6f", "#b37070",
    "#4b0082", "#ffff99", "#FF69B4", "#d950ff", "#e665b4", "#89ccad", "#d1309b"
]
# ────────────────────────────────────────────────────────────────────────────

# 1) Load inputs
for path in (FDR_CSV, SLIDE_MEANS_CSV):
    if not os.path.isfile(path):
        sys.exit(f"ERROR: cannot find required file:\n  {path}")

df    = pd.read_csv(FDR_CSV)
means = pd.read_csv(SLIDE_MEANS_CSV)

if 'cell_type' not in df.columns:
    sys.exit(f"ERROR: 'cell_type' missing from {FDR_CSV}")

# 2) Build color map
cell_order = df['cell_type'].unique().tolist()
color_map  = {ct: PALETTE[i % len(PALETTE)] for i, ct in enumerate(cell_order)}

# 3) Melt slide_means into long form
long_means = means.melt(
    id_vars=["Slide", "Region"],
    value_vars=cell_order,
    var_name="cell_type",
    value_name="mean_fraction"
)

# 4) Volcano for each region-pair
for r1, r2 in REGION_PAIRS:
    print(f"Plotting volcano: {r1} vs {r2} …")
    sub = df[(df.Region1 == r1) & (df.Region2 == r2)].copy()

    # 4a) compute median per cell_type & region
    med = (
        long_means
        .groupby(["cell_type", "Region"])["mean_fraction"]
        .median()
        .reset_index()
    )
    m1 = med[med.Region==r1].set_index("cell_type")["mean_fraction"]
    m2 = med[med.Region==r2].set_index("cell_type")["mean_fraction"]

    # 4b) effect size & p-value transform
    sub["delta"]      = sub["cell_type"].map(m2) - sub["cell_type"].map(m1)
    sub["neg_log10p"] = -np.log10(sub["p_adj_fdr"].replace(0, 1e-300))

    # 5) make the scatter
    fig, ax = plt.subplots(figsize=(14,6))
    for ct in cell_order:
        pts = sub[sub.cell_type == ct]
        ax.scatter(
            pts.delta, pts.neg_log10p,
            color=color_map[ct],
            edgecolors='black',
            linewidths=0.8,
            s=50,  # ensure markers are visible
            zorder=2
        )

    # FDR threshold line
    ax.axhline(-np.log10(0.05), ls='--', c='black', lw=1)

    # region arrow + labels
    lx, ux = ax.get_xlim()
    ly, uy = ax.get_ylim()
    y0 = ly - 0.08 * (uy - ly)
    ax.annotate(r1, xy=(lx, y0), ha='left',  va='center')
    ax.annotate(r2, xy=(ux, y0), ha='right', va='center')
    ax.annotate(
        '', xy=(ux, y0), xytext=(lx, y0),
        arrowprops=dict(arrowstyle='<->', lw=1.5, color='black')
    )

    ax.set_xlabel(f"{r2} median – {r1} median")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title(f"Volcano: {r1} vs {r2}")

    # 6) split & sort by sign of Δ for two side-by-side legends
    delta_series = (
        sub[['cell_type','delta']]
        .drop_duplicates()
        .set_index('cell_type')['delta']
    )

    # negatives (most negative → least negative)
    neg_items = sorted(
        [(ct, d) for ct, d in delta_series.items() if d < 0],
        key=lambda x: x[1]
    )
    neg_cts = [ct for ct, _ in neg_items]
    neg_handles = [
        Line2D([0],[0], marker='o', color=color_map[ct],
               linestyle='', markeredgecolor='black', markersize=6)
        for ct in neg_cts
    ]

    # positives (smallest positive → largest positive)
    pos_items = sorted(
        [(ct, d) for ct, d in delta_series.items() if d >= 0],
        key=lambda x: x[1]
    )
    pos_cts = [ct for ct, _ in pos_items]
    pos_handles = [
        Line2D([0],[0], marker='o', color=color_map[ct],
               linestyle='', markeredgecolor='black', markersize=6)
        for ct in pos_cts
    ]

    # 7) shrink axes to make room on right
    plt.subplots_adjust(right=0.60)

    # 8a) left legend: Δ < 0, titled r1
    leg1 = ax.legend(
        neg_handles, neg_cts,
        title=r1,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.95),
        frameon=False
    )
    ax.add_artist(leg1)

    # 8b) right legend: Δ ≥ 0, titled r2
    leg2 = ax.legend(
        pos_handles, pos_cts,
        title=r2,
        loc="upper left",
        bbox_to_anchor=(1.35, 0.95),
        frameon=False
    )

    # 9) finalize & save
    plt.tight_layout(rect=(0, 0, 0.90, 1))
    out_file = os.path.join(
        OUTDIR,
        f"volcano_{r1.replace(' ','_')}_vs_{r2.replace(' ','_')}.png"
    )
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"  → saved {out_file}\n")


# In[ ]:


# Composition bar plots
import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
ROOT_DIR = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/Normalized_by_broad_type"
SLIDE_MEANS_CSV = os.path.join(ROOT_DIR, "slide_means_normalized_counts.csv")
REGIONS = ["Healthy", "Peritumour", "Tumour"]
TARGET_CELL_TYPES = [
    "Stroma.B.2", "Macrophage", "Plasma", "Plasmablast",
    "Conventional.dendritic.cell",
    "Stroma.C", "Monocyte", "Stroma.D",
    "Plasmacytoid.dendritic.cell", "Mast.cell", "B.cell",
    "T.cell.CD8", "T.cell.CD4", "Stroma.A", "T.cell.regulatory",
    "Neutrophil", "NK.cell"
]

color_map = {
    "Stroma.A": "#355704",
    "Stroma.B.2": "#518508",
    "Stroma.C": "#76ad28",
    "Stroma.D": "#a0db4d",
    "T.cell.CD8": "#540f61",
    "T.cell.CD4": "#81368f",
    "T.cell.regulatory": "#b384bd",
    "Plasma": "#a006bd",
    "B.cell": "#d928fa",
    "Plasmablast": "#e897f7",
    "NK.cell": "#e3cfe8",
    "Mast.cell": "#284a4d",
    "Conventional.dendritic.cell": "#40787d",
    "Plasmacytoid.dendritic.cell": "#51989e",
    "Macrophage": "#62b8bf",
    "Monocyte": "#9bcacf",
    "Neutrophil": "#92e8f0"
}

# Output folder
OUT_PLOT_DIR = os.path.join(ROOT_DIR, "region_composition_barplots")
os.makedirs(OUT_PLOT_DIR, exist_ok=True)

# Read slide-level means (already filtered for exclusions)
slide_means = pd.read_csv(SLIDE_MEANS_CSV)

# Make region-level mean for each cell type (across all slides, per region)
region_means = (
    slide_means.groupby("Region")[TARGET_CELL_TYPES]
    .mean()
    .reindex(REGIONS)
)

# For each region, plot the composition
for region in REGIONS:
    means = region_means.loc[region]
    # Sort for prettier plot (optional)
    means = means.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        means.index,
        means.values,
        color=[color_map[k] for k in means.index]
    )
    ax.set_ylabel("Mean proportion (across slides)")
    ax.set_title(f"Region: {region} — Relative composition of corresponding broad cell_types")
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(means.index, rotation=90, fontsize=7)
    ax.set_ylim(0, 0.6)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    plt.tight_layout()

    out_png = os.path.join(OUT_PLOT_DIR, f"{region}_composition_barplot.png")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved {out_png}")

    


# In[ ]:


# Composition bar plots splits
import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
ROOT_DIR = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/Normalized_by_broad_type"
SLIDE_MEANS_CSV = os.path.join(ROOT_DIR, "slide_means_normalized_counts.csv")
REGIONS = ["Healthy", "Peritumour", "Tumour"]
TARGET_CELL_TYPES = [
    "Stroma.B.2", "Macrophage", "Plasma", "Plasmablast",
    "Conventional.dendritic.cell",
    "Stroma.C", "Monocyte", "Stroma.D",
    "Plasmacytoid.dendritic.cell", "Mast.cell", "B.cell",
    "T.cell.CD8", "T.cell.CD4", "Stroma.A", "T.cell.regulatory",
    "Neutrophil", "NK.cell"
]

color_map = {
    "Stroma.A": "#355704",
    "Stroma.B.2": "#518508",
    "Stroma.C": "#76ad28",
    "Stroma.D": "#a0db4d",
    "T.cell.CD8": "#540f61",
    "T.cell.CD4": "#81368f",
    "T.cell.regulatory": "#b384bd",
    "Plasma": "#a006bd",
    "B.cell": "#d928fa",
    "Plasmablast": "#e897f7",
    "NK.cell": "#e3cfe8",
    "Mast.cell": "#284a4d",
    "Conventional.dendritic.cell": "#40787d",
    "Plasmacytoid.dendritic.cell": "#51989e",
    "Macrophage": "#62b8bf",
    "Monocyte": "#9bcacf",
    "Neutrophil": "#92e8f0"
}

# === Map each cell type to a broad type ===
broad_type_map = {
    # CAFs
    "Stroma.A": "CAFs",
    "Stroma.B.2": "CAFs",
    "Stroma.C": "CAFs",
    "Stroma.D": "CAFs",
    # Immune
    "T.cell.CD8": "Immune Cells",
    "T.cell.CD4": "Immune Cells",
    "T.cell.regulatory": "Immune Cells",
    "Plasma": "Immune Cells",
    "B.cell": "Immune Cells",
    "Plasmablast": "Immune Cells",
    "NK.cell": "Immune Cells",
    "Mast.cell": "Immune Cells",
    "Conventional.dendritic.cell": "Immune Cells",
    "Plasmacytoid.dendritic.cell": "Immune Cells",
    "Macrophage": "Immune Cells",
    "Monocyte": "Immune Cells",
    "Neutrophil": "Immune Cells"
}

broad_types = ["Immune Cells", "CAFs"]

# Output base folder
OUT_PLOT_BASE = os.path.join(ROOT_DIR, "region_composition_barplots")
os.makedirs(OUT_PLOT_BASE, exist_ok=True)

# Read slide-level means (already filtered for exclusions)
slide_means = pd.read_csv(SLIDE_MEANS_CSV)

# Region-level mean for each cell type
region_means = (
    slide_means.groupby("Region")[TARGET_CELL_TYPES]
    .mean()
    .reindex(REGIONS)
)

# For each region and broad type, plot the bars
for region in REGIONS:
    means = region_means.loc[region]
    for broad_type in broad_types:
        # Filter cell types for this broad type
        ct_list = [ct for ct in TARGET_CELL_TYPES if broad_type_map.get(ct) == broad_type]
        if not ct_list:
            continue

        # Sort for prettier plot
        sub_means = means[ct_list].sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(
            sub_means.index,
            sub_means.values,
            color=[color_map[k] for k in sub_means.index]
        )
        ax.set_ylabel("Mean proportion (across slides)")
        ax.set_title(f"{region}: {broad_type} composition")
        ax.set_xticks(range(len(sub_means)))
        ax.set_xticklabels(sub_means.index, rotation=90, fontsize=7)
        ax.set_ylim(0, 0.6)
        ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        plt.tight_layout()

        # Make subfolder for region
        region_dir = os.path.join(OUT_PLOT_BASE, region)
        os.makedirs(region_dir, exist_ok=True)
        out_png = os.path.join(region_dir, f"{broad_type}_barplot.png")
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"Saved {out_png}")


# In[ ]:


get_ipython().run_cell_magic('writefile', 'Combine0607_broad_cell_type_region_level_counts.py', 'import os\nimport glob\nimport pandas as pd\n\n# ─── EDIT ONLY THIS ───────────────────────────────────────────────────\n# Base folder containing per‐ROI normalized counts:\nROOT_NORM = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/Normalized_by_broad_type"\n# ──────────────────────────────────────────────────────────────────────\n\ndef combine_region(region_dir):\n    """\n    For a given region folder, find all ROI_*_by_broadtype.csv under any Slide_* subfolder,\n    concatenate them into a single DataFrame, save that CSV in the region folder,\n    and print the number of columns excluding the three metadata columns.\n    """\n    region_name = os.path.basename(region_dir.rstrip("/"))\n    all_csvs = glob.glob(os.path.join(region_dir, "Slide_*", "*", "*_by_broadtype.csv"), recursive=False)\n\n    if not all_csvs:\n        print(f"[{region_name}] No ROI CSV files found under {region_dir!r}.")\n        return\n\n    df_list = []\n    for fp in sorted(all_csvs):\n        df = pd.read_csv(fp)\n        df_list.append(df)\n\n    # Concatenate all per‐ROI DataFrames (they share the same columns)\n    combined = pd.concat(df_list, axis=0, ignore_index=True, sort=False).fillna(0)\n\n    # Save combined CSV into the region folder\n    out_csv = os.path.join(region_dir, f"{region_name}_all_ROIs_normalized_counts.csv")\n    combined.to_csv(out_csv, index=False)\n\n    # Count columns excluding the three metadata columns: Region, Slide, ROI\n    total_cols = combined.shape[1]\n    metadata_cols = [c for c in ("Region", "Slide", "ROI") if c in combined.columns]\n    num_data_cols = total_cols - len(metadata_cols)\n\n    print(f"[{region_name}] Combined {len(df_list)} ROI files → saved \'{out_csv}\'.")\n    print(f"[{region_name}] Number of columns (excluding {metadata_cols}): {num_data_cols}\\n")\n\n\ndef main():\n    if not os.path.isdir(ROOT_NORM):\n        raise FileNotFoundError(f"Root folder not found: {ROOT_NORM!r}")\n\n    # List all immediate subdirectories (each should be a region)\n    regions = [\n        os.path.join(ROOT_NORM, d) for d in os.listdir(ROOT_NORM)\n        if os.path.isdir(os.path.join(ROOT_NORM, d))\n    ]\n    if not regions:\n        print(f"No region folders found under {ROOT_NORM!r}.")\n        return\n\n    for region_dir in sorted(regions):\n        combine_region(region_dir)\n\n\nif __name__ == "__main__":\n    main()\n')


# In[ ]:





# In[ ]:


# 2.1.1 umap


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

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
ROOT       = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/Normalized_cell_count"
REGIONS    = ["Tumour", "Peritumour", "Healthy"]
SUMMARY_FN = "{region}_all_ROIs_normalized_counts.csv"

# where to dump your PCA/UMAP outputs:
OUT_DIR = os.path.join(ROOT, "All_cell_types_PCA_UMAP_with_CLR_ComBat")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 1) READ, ANNOTATE & CONCATENATE ──────────────────────────────────────────
dfs = []
for region in REGIONS:
    in_path = os.path.join(ROOT, region, SUMMARY_FN.format(region=region))
    df = pd.read_csv(in_path).fillna(0.0)
    df["Region"] = region
    keep = ["Region","Slide","ROI"] + [c for c in df.columns 
                                        if c not in ("Region","Slide","ROI")]
    df_sub = df[keep]
    out_csv = os.path.join(OUT_DIR, f"{region}_all_celltypes_normalized_counts.csv")
    df_sub.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv!r}")
    dfs.append(df_sub)

big_df = pd.concat(dfs, ignore_index=True)
big_df = big_df.fillna(0.0)
combined_csv = os.path.join(OUT_DIR, "all_regions_all_celltypes_normalized_counts.csv")
big_df.to_csv(combined_csv, index=False)
print(f"Wrote combined → {combined_csv!r}")

big_df_idx   = big_df.set_index(["Region","Slide","ROI"])
feature_cols = list(big_df_idx.columns)
X_raw        = big_df_idx[feature_cols].values


# ─── 2) CLR TRANSFORM + COMBAT + SCALE ────────────────────────────────────────
eps    = 1e-6
X_safe = X_raw + eps
logX   = np.log(X_safe)
X_clr  = logX - logX.mean(axis=1, keepdims=True)

adata = sc.AnnData(X_clr)
adata.obs['batch']  = big_df_idx.index.get_level_values("Slide")
adata.obs['region'] = big_df_idx.index.get_level_values("Region")

# Run ComBat from sc.pp, preserving region
sc.pp.combat(adata, key='batch', covariates=['region'])

X_corr = adata.X
Xs     = StandardScaler().fit_transform(X_corr)


# ─── 3) PCA ───────────────────────────────────────────────────────────────────
pca    = PCA(n_components=2, random_state=0)
coords = pca.fit_transform(Xs)
loadings = pd.DataFrame(pca.components_.T,
                        index=feature_cols,
                        columns=["PC1","PC2"])
explained = pd.Series(pca.explained_variance_ratio_,
                      index=["PC1","PC2"],
                      name="explained_variance_ratio")

coords_df = pd.DataFrame(coords,
                         columns=["PC1","PC2"],
                         index=big_df_idx.index)

coords_df.reset_index().to_csv(os.path.join(OUT_DIR,"PCA_coords.csv"), index=False)
loadings.to_csv(os.path.join(OUT_DIR,"PCA_loadings.csv"))
explained.to_csv(os.path.join(OUT_DIR,"PCA_explained_variance.csv"))
print("Saved PCA coords, loadings, explained-variance")


# ─── 4) Plot PCA by Region ───────────────────────────────────────────────────
color_map = {"Tumour":"orange","Peritumour":"lightblue","Healthy":"lightgreen"}
pca_png = os.path.join(OUT_DIR,"PCA_plot_by_region.png")

plt.figure(figsize=(7,6))
for region in REGIONS:
    mask = coords_df.index.get_level_values("Region")==region
    plt.scatter(coords[mask,0], coords[mask,1],
                label=region, s=12, alpha=0.7, color=color_map[region])

handles, labels = plt.gca().get_legend_handles_labels()
order = [labels.index(r) for r in REGIONS]
plt.legend([handles[i] for i in order],
           [labels[i] for i in order],
           bbox_to_anchor=(1.02,1), loc="upper left",
           frameon=False, title="Region")

plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("PCA (all cell types)")
plt.tight_layout(rect=[0.8,0.8,1,1])
plt.savefig(pca_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"PCA plot → {pca_png}")


# ─── 5) UMAP ──────────────────────────────────────────────────────────────────
reducer = umap.UMAP(n_neighbors=50, min_dist=0.5, spread=2, random_state=42)
Z       = reducer.fit_transform(Xs)
umap_df = pd.DataFrame(Z,
                       columns=["UMAP1","UMAP2"],
                       index=big_df_idx.index)
umap_df.reset_index().to_csv(os.path.join(OUT_DIR,"UMAP_coords.csv"), index=False)
print("Saved UMAP coords")

# ─── 6) Plot UMAP by Region ──────────────────────────────────────────────────
umap_png = os.path.join(OUT_DIR,"UMAP_plot_by_region.png")
fig, ax = plt.subplots(figsize=(7,6))
for region in REGIONS:
    mask = umap_df.index.get_level_values("Region")==region
    ax.scatter(Z[mask,0], Z[mask,1],
               label=region, s=12, alpha=0.7,
               edgecolors="none", color=color_map[region])

handles, labels = ax.get_legend_handles_labels()
order = [labels.index(r) for r in REGIONS]
ax.legend([handles[i] for i in order],
          [labels[i] for i in order],
          bbox_to_anchor=(1.02,1), loc="upper left",
          frameon=False, title="Region")

ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
ax.set_title("UMAP (all cell types)")
plt.tight_layout(rect=[0.8,0.8,1,1])
fig.savefig(umap_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"UMAP plot → {umap_png}")

# ─── SLIDE COLOR MAP ────────────────────────────────────────────────────────────
# 1) define the full, ordered list of slides you have in your project
SLIDES = ["Slide_1","Slide_2","Slide_3","Slide_4"]  # ← adjust to your real slide IDs, in sorted order

# 2) pick a colormap with enough distinct colours
cmap = plt.get_cmap("tab20")

# 3) build a mapping from slide → RGBA colour
SLIDE_COLORS = {s: cmap(i % cmap.N) for i, s in enumerate(SLIDES)}
# ────────────────────────────────────────────────────────────────────────────────


# ─── 7) PLOT PCA (colored by Slide) ───────────────────────────────────────────
# build a color map for slides
slides = sorted(big_df_idx.index.get_level_values("Slide").unique())
cmap = plt.get_cmap("tab20")
slide_colors = dict(zip(slides, cmap(np.linspace(0, 1, len(slides)))))

pca_slide_png = os.path.join(OUT_DIR, "PCA_plot_by_slide.png")
plt.figure(figsize=(7,6))
# use the global SLIDES and SLIDE_COLORS
for slide in SLIDES:
    mask = coords_df.index.get_level_values("Slide") == slide
    plt.scatter(
        coords[mask,0], coords[mask,1],
        label=slide,
        s=12, alpha=0.7,
        color=SLIDE_COLORS[slide]
    )
plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=False,
    title="Slide",
    fontsize="small"
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA (all cell types) colored by Slide")
plt.tight_layout(rect=[0.8,0.8,1,1])
plt.savefig(pca_slide_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"PCA by Slide → {pca_slide_png}")


# ─── 8) PLOT UMAP (colored by Slide) ──────────────────────────────────────────
umap_slide_png = os.path.join(OUT_DIR, "UMAP_plot_by_slide.png")
fig, ax = plt.subplots(figsize=(7,6))
for slide in SLIDES:
    mask = umap_df.index.get_level_values("Slide") == slide
    ax.scatter(
        Z[mask,0], Z[mask,1],
        label=slide,
        s=12, alpha=0.7,
        color=SLIDE_COLORS[slide],
        edgecolors="none"
    )
ax.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=False,
    title="Slide",
    fontsize="small"
)
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.set_title("UMAP (all cell types) colored by Slide")
plt.tight_layout(rect=[0.8,0.8,1,1])
fig.savefig(umap_slide_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"UMAP by Slide → {umap_slide_png}")


# In[ ]:


# Targeted Cell types.


# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

# for CLR + ComBat
import scanpy as sc

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
ROOT       = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/Normalized_cell_count"
REGIONS    = ["Tumour", "Peritumour", "Healthy"]
SUMMARY_FN = "{region}_all_ROIs_normalized_counts.csv"

# custom colors for each region
color_map = {
    "Tumour":     "orange",
    "Peritumour": "lightblue",
    "Healthy":    "lightgreen"
}

# where to dump your PCA/UMAP outputs:
OUT_DIR = os.path.join(ROOT, "Target_cell_types_PCA_UMAP_ComBat")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 1) load, filter & save per‐region ─────────────────────────────────────────
TARGET_CELL_TYPES = [
    "Stroma.B.2", "Macrophage", "Endothelial.B", "Plasma", "Plasmablast",
    "Conventional.dendritic.cell", "Periportal.LSEC",
    "Stroma.C", "Monocyte", "Stroma.D", "Endothelial.A",
    "Plasmacytoid.dendritic.cell", "Mast.cell", "B.cell",
    "T.cell.CD8", "T.cell.CD4", "Stroma.A", "T.cell.regulatory",
    "Neutrophil", "NK.cell","Central.venous.LSEC"
]
INPUT_META = ["Slide", "ROI"]

dfs = []
counts = []              # to store per-region counts
total_rois_all  = 0
kept_rois_all   = 0

for region in REGIONS:
    in_path = os.path.join(ROOT, region, SUMMARY_FN.format(region=region))
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Can't find {in_path!r}")
    df = pd.read_csv(in_path)
    keep = INPUT_META + TARGET_CELL_TYPES
    missing = set(TARGET_CELL_TYPES) - set(df.columns)
    if missing:
        raise KeyError(f"File {in_path!r} missing: {missing}")
    df = df[keep].fillna(0.0)

    # filter to ROIs with any targeted cell type > 0
    total_rois_region = len(df)
    mask_any = df[TARGET_CELL_TYPES].sum(axis=1) > 0
    kept_rois_region = int(mask_any.sum())
    df = df[mask_any].copy()

    total_rois_all += total_rois_region
    kept_rois_all  += kept_rois_region
    counts.append({
        "Region": region,
        "ROIs_total": total_rois_region,
        "ROIs_kept":  kept_rois_region
    })
    print(f"{region}: kept {kept_rois_region}/{total_rois_region} ROIs")

    df["Region"] = region
    out_region_csv = os.path.join(
        OUT_DIR,
        f"{region}_filtered_{len(TARGET_CELL_TYPES)}celltypes_normalized_counts.csv"
    )
    df.to_csv(out_region_csv, index=False)
    print(f"Wrote filtered → {out_region_csv!r}")
    dfs.append(df)

# save counts summary
counts_df = pd.DataFrame(counts + [{
    "Region": "ALL",
    "ROIs_total": total_rois_all,
    "ROIs_kept":  kept_rois_all
}])
counts_csv = os.path.join(OUT_DIR, "ROI_inclusion_counts.csv")
counts_df.to_csv(counts_csv, index=False)
print(f"ROI inclusion counts → {counts_csv!r}")
print(f"TOTAL: kept {kept_rois_all}/{total_rois_all} ROIs")

# ─── 2) concatenate & CLR → ComBat → scale ────────────────────────────────────
big_df = pd.concat(dfs, ignore_index=True)
big_df = big_df.set_index(["Region"] + INPUT_META)

# raw fraction matrix
X_raw = big_df[TARGET_CELL_TYPES].values

# wrap in AnnData for ComBat (no CLR)
adata = sc.AnnData(X_raw)
adata.obs['batch']  = big_df.index.get_level_values("Slide")
adata.obs['region'] = big_df.index.get_level_values("Region")
# remove slide, preserve region
sc.pp.combat(adata, key='batch', covariates=['region'])

# standardized corrected matrix
X_corr = adata.X
Xs     = StandardScaler().fit_transform(X_corr)

# save combined filtered table
combined_csv = os.path.join(OUT_DIR, "all_regions_filtered_normalized_counts.csv")
big_df.reset_index()[["Region"] + INPUT_META + TARGET_CELL_TYPES] \
     .to_csv(combined_csv, index=False)
print(f"Saved combined filtered table → {combined_csv!r}")

# ─── SLIDE COLOR MAP ────────────────────────────────────────────────────────────
# 1) define the full, ordered list of slides you have in your project
SLIDES = ["Slide_1","Slide_2","Slide_3","Slide_4"]  # ← adjust to your real slide IDs, in sorted order

# 2) pick a colormap with enough distinct colours
cmap = plt.get_cmap("tab20")

# 3) build a mapping from slide → RGBA colour
SLIDE_COLORS = {s: cmap(i % cmap.N) for i, s in enumerate(SLIDES)}
# ────────────────────────────────────────────────────────────────────────────────

# ─── 3) PCA ───────────────────────────────────────────────────────────────────
pca    = PCA(n_components=2, random_state=0)
coords = pca.fit_transform(Xs)
loadings = pd.DataFrame(pca.components_.T,
                        index=TARGET_CELL_TYPES,
                        columns=["PC1","PC2"])
explained = pd.Series(pca.explained_variance_ratio_,
                      index=["PC1","PC2"],
                      name="explained_variance_ratio")

coords_df = pd.DataFrame(coords,
                         columns=["PC1","PC2"],
                         index=big_df.index)
coords_df.to_csv(os.path.join(OUT_DIR, "all_regions_PCA_coords.csv"))
loadings.to_csv(os.path.join(OUT_DIR, "all_regions_PCA_loadings.csv"))
explained.to_csv(os.path.join(OUT_DIR, "all_regions_PCA_explained.csv"))
print("Saved PCA outputs")

# ─── 4) Plot PCA (by Region) ─────────────────────────────────────────────────
pca_png = os.path.join(OUT_DIR, "all_regions_PCA_by_region.png")
plt.figure(figsize=(7,6))
for region in REGIONS:
    m = coords_df.index.get_level_values("Region")==region
    plt.scatter(coords[m,0], coords[m,1],
                label=region, s=12, alpha=0.7, color=color_map[region])
handles, labels = plt.gca().get_legend_handles_labels()
order = [labels.index(r) for r in REGIONS]
plt.legend([handles[i] for i in order],
           [labels[i] for i in order],
           bbox_to_anchor=(1.02,1), loc="upper left",
           frameon=False, title="Region")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("PCA (targeted cell‐type counts)")
plt.tight_layout(rect=[0.8,0.8,1,1])
plt.savefig(pca_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"PCA plot → {pca_png}")

# ─── 4b) Plot PCA (by Slide) ─────────────────────────────────────────────────
pca_slide_png = os.path.join(OUT_DIR, "all_regions_PCA_by_slide_tab20.png")
slides = coords_df.index.get_level_values("Slide").unique()
cmap   = plt.get_cmap("tab20")
slide_colors = {s: cmap(i % 20) for i,s in enumerate(slides)}

plt.figure(figsize=(7,6))
# use the global SLIDES and SLIDE_COLORS
for slide in SLIDES:
    mask = coords_df.index.get_level_values("Slide") == slide
    plt.scatter(
        coords[mask,0], coords[mask,1],
        label=slide,
        s=12, alpha=0.7,
        color=SLIDE_COLORS[slide]
    )
plt.legend(bbox_to_anchor=(1.02,1), loc="upper left",
           frameon=False, title="Slide", fontsize="small")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("PCA (colored by Slide)")
plt.tight_layout(rect=[0.8,0.8,1,1])
plt.savefig(pca_slide_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"PCA-by-Slide → {pca_slide_png}")

# ─── 5) UMAP ─────────────────────────────────────────────────────────────────
reducer = umap.UMAP(n_neighbors=50, min_dist=0.5, spread=2, random_state=42)
Z       = reducer.fit_transform(Xs)
umap_df = pd.DataFrame(Z, columns=["UMAP1","UMAP2"], index=big_df.index)
umap_df.to_csv(os.path.join(OUT_DIR, "all_regions_UMAP_coords.csv"))
print("Saved UMAP coords")

# ─── 6) Plot UMAP (by Region) ────────────────────────────────────────────────
umap_png = os.path.join(OUT_DIR, "all_regions_UMAP_by_region.png")
fig, ax = plt.subplots(figsize=(7,6))
for region in REGIONS:
    m = umap_df.index.get_level_values("Region")==region
    ax.scatter(Z[m,0], Z[m,1],
               label=region, s=12, alpha=0.7,
               color=color_map[region], edgecolors="none")
handles, labels = ax.get_legend_handles_labels()
order = [labels.index(r) for r in REGIONS]
ax.legend([handles[i] for i in order],
          [labels[i] for i in order],
          bbox_to_anchor=(1.02,1), loc="upper left",
          frameon=False, title="Region")
ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
ax.set_title("UMAP (targeted cell‐type counts)")
plt.tight_layout(rect=[0.8,0.8,1,1])
fig.savefig(umap_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"UMAP plot → {umap_png}")

# ─── 6b) Plot UMAP (by Slide) ────────────────────────────────────────────────
umap_slide_png = os.path.join(OUT_DIR, "all_regions_UMAP_by_slide_tab20.png")
fig, ax = plt.subplots(figsize=(7,6))
for slide in SLIDES:
    mask = umap_df.index.get_level_values("Slide") == slide
    ax.scatter(
        Z[mask,0], Z[mask,1],
        label=slide,
        s=12, alpha=0.7,
        color=SLIDE_COLORS[slide],
        edgecolors="none"
    )
ax.legend(bbox_to_anchor=(1.02,1), loc="upper left",
          frameon=False, title="Slide", fontsize="small")
ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
ax.set_title("UMAP (colored by Slide)")
plt.tight_layout(rect=[0.8,0.8,1,1])
fig.savefig(umap_slide_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"UMAP-by-Slide → {umap_slide_png}")


# In[ ]:


# 2.1.2 umap
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import scanpy as sc

# ==== CONFIGURATION ====
ROOT = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/Normalized_by_broad_type"
REGIONS = ["Tumour", "Peritumour", "Healthy"]
SUMMARY_FN = "{region}_all_ROIs_normalized_counts.csv"
OUT_BASE = os.path.join(ROOT, "Target_cell_types_PCA_UMAP_FILTERED_ComBat")

# === Map each cell_type to its broad_type ===
broad_map = {}
for ct in ["Stroma.A", "Stroma.B.2", "Stroma.C", "Stroma.D"]:
    broad_map[ct] = "CAFs"
for ct in [
    "Macrophage", "Plasma", "Plasmablast",
    "Conventional.dendritic.cell", "Monocyte",
    "Plasmacytoid.dendritic.cell", "Mast.cell", "B.cell",
    "T.cell.CD8", "T.cell.CD4", "T.cell.regulatory",
    "Neutrophil", "NK.cell"
]:
    broad_map[ct] = "Immune Cells"

# All cell types (useful for loading columns)
ALL_CELL_TYPES = list(broad_map.keys())
BROAD_TYPES = sorted(set(broad_map.values()))

INPUT_META = ["Slide", "ROI"]

# --- Function to load and filter per-broad type ---
def load_and_filter_data(broad_type):
    dfs = []
    for region in REGIONS:
        in_path = os.path.join(ROOT, region, SUMMARY_FN.format(region=region))
        if not os.path.isfile(in_path):
            raise FileNotFoundError(f"Can't find {in_path!r}")
        df = pd.read_csv(in_path)
        # Only keep meta + cell types for this broad type
        feats = [ct for ct in ALL_CELL_TYPES if broad_map[ct] == broad_type]
        missing = set(feats) - set(df.columns)
        if missing:
            raise KeyError(f"{in_path!r} missing: {missing}")
        df = df[INPUT_META + feats].fillna(0.0)
        df["Region"] = region
        dfs.append(df)
    big_df = pd.concat(dfs, ignore_index=True)
    big_df[["Slide", "ROI"]] = big_df[["Slide", "ROI"]].astype(str)
    return big_df

# --- Function to apply region-wise exclusions ---
EXCL_ROOT = os.path.dirname(ROOT)

def apply_exclusions(big_df, region, broad_type):
    excl_fn = f"Exclude_{region}_rois.csv"
    excl_path = os.path.join(EXCL_ROOT, excl_fn)
    mask = pd.Series([True] * len(big_df), index=big_df.index)
    excl_set = set()
    n_excluded = 0
    if os.path.isfile(excl_path):
        excl = pd.read_csv(excl_path, dtype=str)
        excl = excl[excl["broad_type"] == broad_type]
        if excl.empty:
            print(f"No exclusions for {broad_type} in {region}.")
        else:
            excl_set = set(tuple(x) for x in excl[["Slide", "ROI"]].values)
            n_excluded = len(excl_set)
            print(f"{broad_type} in {region}: excluding {n_excluded} unique ROIs.")
        mask = ~big_df.apply(
            lambda r: r["Region"] == region and (r["Slide"], r["ROI"]) in excl_set,
            axis=1
        )
    else:
        print(f"Warning: No exclusion list found for {region}, keeping all ROIs in this region.")
    return mask, excl_set

# --- Run PCA/UMAP/ComBat for each broad type ---
for broad_type in BROAD_TYPES:
    print(f"\n=== {broad_type} ===")
    OUT_DIR = os.path.join(OUT_BASE, broad_type.replace(" ", "_"))
    os.makedirs(OUT_DIR, exist_ok=True)
    feats = [ct for ct in ALL_CELL_TYPES if broad_map[ct] == broad_type]
    big_df = load_and_filter_data(broad_type)
    
    # Track all unique exclusions
    all_excluded = set()
    region_masks = []
    for region in REGIONS:
        region_mask, excl_set = apply_exclusions(big_df, region, broad_type)
        all_excluded.update(excl_set)
        region_masks.append(region_mask)
    final_mask = np.logical_and.reduce(region_masks)
    filtered = big_df[final_mask].copy()
    
    print(f"TOTAL unique excluded ROIs for {broad_type}: {len(all_excluded)}")
    filtered.to_csv(os.path.join(OUT_DIR, "all_regions_filtered_counts.csv"), index=False)
    big_df.to_csv(os.path.join(OUT_DIR, "all_regions_unfiltered_counts.csv"), index=False)
    # Set index for embedding
    filtered.set_index(["Region", "Slide", "ROI"], inplace=True)
        # ─── ComBat batch correction ───────────────────────────────────────────────
    # 1) Build AnnData on raw features
    adata = sc.AnnData(filtered[feats].values.astype(np.float32))

    # 2) add batch column
    adata.obs['batch'] = filtered.index.get_level_values('Slide').astype(str).values

    # 3) one-hot encode Region covariate (drop_first to avoid collinearity)
    region_idx = filtered.index.get_level_values('Region')
    region_dummies = pd.get_dummies(region_idx, prefix='region', drop_first=True)
    for col in region_dummies.columns:
        adata.obs[col] = region_dummies[col].values.astype(np.float32)

    # 4) run ComBat, preserving Region effects
    sc.pp.combat(
        adata,
        key='batch',
        covariates=region_dummies.columns.tolist()
    )

    # 5) pull out corrected matrix and re-scale for PCA/UMAP
    X_corr = adata.X
    Xs     = StandardScaler().fit_transform(X_corr)
    # ──────────────────────────────────────────────────────────────────────────
    # --- PCA ---
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(Xs)
    coords_df = pd.DataFrame(coords, columns=["PC1", "PC2"], index=filtered.index)
    coords_df.to_csv(os.path.join(OUT_DIR, "PCA_coords.csv"))
    print("Saved PCA coords")
    # --- Plot PCA by Region ---
    color_map = {
        "Tumour": "orange",
        "Peritumour": "lightblue",
        "Healthy": "lightgreen"
    }
    pca_png = os.path.join(OUT_DIR, "PCA_by_region.png")
    plt.figure(figsize=(7,6))
    for r in REGIONS:
        m = coords_df.index.get_level_values("Region") == r
        plt.scatter(coords[m,0], coords[m,1],
                    label=r, color=color_map[r], s=10, alpha=0.7)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [labels.index(r) for r in REGIONS]
    plt.legend([handles[i] for i in order], [labels[i] for i in order],
               bbox_to_anchor=(1.02,1), loc="upper left", frameon=False, title="Region")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"PCA ({broad_type})")
    plt.tight_layout(rect=[0.8,0.8,1,1])
    plt.savefig(pca_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"PCA plot → {pca_png}")
    # --- Plot PCA by Slide ---
    pca_slide_png = os.path.join(OUT_DIR, "PCA_by_slide.png")
    slides = coords_df.index.get_level_values("Slide").unique()
    cmap = plt.get_cmap("tab20")(np.linspace(0,1,len(slides)))
    slide_colors = dict(zip(slides, cmap))
    plt.figure(figsize=(7,6))
    for s in slides:
        m = coords_df.index.get_level_values("Slide") == s
        plt.scatter(coords[m,0], coords[m,1],
                    label=s, color=slide_colors[s], s=10, alpha=0.7)
    plt.legend(bbox_to_anchor=(1.02,1), loc="upper left",
               frameon=False, title="Slide", fontsize="small")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"PCA ({broad_type} by Slide)")
    plt.tight_layout(rect=[0.8,0.8,1,1])
    plt.savefig(pca_slide_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"PCA by Slide → {pca_slide_png}")
    # --- UMAP ---
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.5, spread=1, random_state=42)
    Z = reducer.fit_transform(Xs)
    umap_df = pd.DataFrame(Z, columns=["UMAP1", "UMAP2"], index=filtered.index)
    umap_df.to_csv(os.path.join(OUT_DIR, "UMAP_coords.csv"))
    print("Saved UMAP coords")
    # --- Plot UMAP by Region ---
    umap_png = os.path.join(OUT_DIR, "UMAP_by_region.png")
    fig, ax = plt.subplots(figsize=(7,6))
    for r in REGIONS:
        m = umap_df.index.get_level_values("Region") == r
        ax.scatter(Z[m,0], Z[m,1],
                   label=r, color=color_map[r], s=12, alpha=0.7, edgecolors="none")
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(r) for r in REGIONS]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              bbox_to_anchor=(1.02,1), loc="upper left", frameon=False, title="Region")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title(f"UMAP ({broad_type})")
    plt.tight_layout(rect=[0.8,0.8,1,1])
    fig.savefig(umap_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"UMAP plot → {umap_png}")
    # --- Plot UMAP by Slide ---
    umap_slide_png = os.path.join(OUT_DIR, "UMAP_by_slide.png")
    fig, ax = plt.subplots(figsize=(7,6))
    for s in slides:
        m = umap_df.index.get_level_values("Slide") == s
        ax.scatter(Z[m,0], Z[m,1],
                   label=s, color=slide_colors[s], s=12, alpha=0.7, edgecolors="none")
    ax.legend(bbox_to_anchor=(1.02,1), loc="upper left", frameon=False,
              title="Slide", fontsize="small")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title(f"UMAP ({broad_type} by Slide)")
    plt.tight_layout(rect=[0.8,0.8,1,1])
    fig.savefig(umap_slide_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"UMAP by Slide → {umap_slide_png}")

print("\nAll done!")


# In[ ]:





# In[ ]:




