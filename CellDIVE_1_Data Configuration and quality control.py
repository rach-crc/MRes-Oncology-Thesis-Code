#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

# path to your original file
infile  = "/Volumes/My Passport/Spatial_Proteomics_data_parent/roi_cells_combined_box500.csv"
# path for the new file
outfile = "/Volumes/My Passport/Spatial_Proteomics_data_final/cell_metadata_muspan_combined_rois_all.csv"

# load
df = pd.read_csv(infile)

# rename the three columns
df = df.rename(columns={
    "phenotype_x":   "broad_type",
    "phenotype_y":   "cell_type",
    "Tissue_Index":  "slide_ID",
    "ROI_category":  "region",
    "X_um":          "x_um",
    "Y_um":          "y_um"
})

# print total number of cells (rows)
print("Total cells:", df.shape[0])

# save
df.to_csv(outfile, index=False)

print(f"Saved renamed table to {outfile}")


# In[ ]:


import pandas as pd

# load your already‐renamed table
df = pd.read_csv("/Volumes/My Passport/Spatial_Proteomics_data_final/cell_metadata_muspan_combined_rois_all.csv")

# group by region and count distinct ROI_IDs
roi_counts = df.groupby("region")["ROI_ID"].nunique().reset_index(name="n_unique_ROIs")

print(roi_counts)


# In[ ]:


import pandas as pd

# load your table
df = pd.read_csv("/Volumes/My Passport/Spatial_Proteomics_data_final/cell_metadata_muspan_combined_rois_all.csv")

# group by broad_type and collect unique cell_types
unique_cell_types = df.groupby("broad_type")["cell_type"].unique()

# display
for btype, ctypes in unique_cell_types.items():
    print(f"{btype}:")
    for ct in ctypes:
        print(f"  - {ct}")


# In[ ]:


# 1) load
fn = "/Volumes/My Passport/Spatial_Proteomics_data_final/cell_metadata_muspan_combined_rois_all.csv"
df = pd.read_csv(fn)

# 2) define your recoding map
recode = {
    "CAFs_FAP_ACtA":      "CAFs",    # your exact spelling may vary
    "lymphatic CAFs":     "CAFs",
    "Macrophages":        "Immune Cells",
    "Neutrophils":        "Immune Cells",
    "T cells":            "Immune Cells",
    "Other Immune cells": "Immune Cells"
}

# 3) apply it (all other broad_types stay unchanged)
df["broad_type"] = df["broad_type"].replace(recode)

# 4) (optional) double-check
print(df["broad_type"].value_counts())

# 5) save back out
out = "/Volumes/My Passport/Spatial_Proteomics_data_final/cell_metadata_muspan_combined_rois_all.csv"
df.to_csv(out, index=False)
print("wrote →", out)


# In[ ]:


import pandas as pd
import os

# load the merged table
df = pd.read_csv("/Volumes/My Passport/Spatial_Proteomics_data_final/cell_metadata_muspan_combined_rois_all.csv")

# output directory
out_dir = "/Volumes/My Passport/Spatial_Proteomics_data_final"
os.makedirs(out_dir, exist_ok=True)

# iterate over each region
for region in df["region"].unique():
    # make a filesystem-friendly name
    clean_reg = region.replace(" ", "_")
    # filter
    sub = df[df["region"] == region]
    # build path
    out_path = os.path.join(
        out_dir,
        f"cell_metadata_muspan_{clean_reg}.csv"
    )
    # save
    sub.to_csv(out_path, index=False)
    print(f"→ {region}: {len(sub)} rows → {out_path}")


# In[ ]:


# Create region folders, slide folders, and ROI csvs.


# In[ ]:


import pandas as pd
import os

# map regions to their pre-split files
region_files = {
    "Tumour":     "/Volumes/My Passport/Spatial_Proteomics_data_final/cell_metadata_muspan_Tumour.csv",
    "Peritumour": "/Volumes/My Passport/Spatial_Proteomics_data_final/cell_metadata_muspan_Peritumour.csv",
    "Healthy":    "/Volumes/My Passport/Spatial_Proteomics_data_final/cell_metadata_muspan_Healthy.csv",
}

base_dir = "/Volumes/My Passport/Spatial_Proteomics_data_final"

for region, infile in region_files.items():
    # load
    df = pd.read_csv(infile)
    
    # clean region name and make region folder
    region_clean = region.replace(" ", "_")
    region_dir   = os.path.join(base_dir, region_clean)
    os.makedirs(region_dir, exist_ok=True)
    
    # for each slide in this region
    for slide in df["slide_ID"].unique():
        slide_dir = os.path.join(region_dir, f"Slide_{slide}")
        os.makedirs(slide_dir, exist_ok=True)
        
        # subset to this slide
        df_slide = df[df["slide_ID"] == slide]
        
        # and for each ROI in that slide, write ROI_{ID}.csv
        for roi in df_slide["ROI_ID"].unique():
            df_roi   = df_slide[df_slide["ROI_ID"] == roi]
            out_file = os.path.join(slide_dir, f"ROI_{roi}.csv")
            df_roi.to_csv(out_file, index=False)
            print(f"Wrote {len(df_roi)} rows → {out_file}")


# In[ ]:


import os
import pandas as pd

# paths
base_dir      = "/Volumes/My Passport/Spatial_Proteomics_data_final"
combined_dir  = os.path.join(base_dir, "Combined")
combined_file = os.path.join(base_dir, "cell_metadata_muspan_combined_rois_all.csv")

# make sure the Combined folder exists
os.makedirs(combined_dir, exist_ok=True)

# load the full table
df = pd.read_csv(combined_file)

# for each slide
for slide in df["slide_ID"].unique():
    slide_dir = os.path.join(combined_dir, f"Slide_{slide}")
    os.makedirs(slide_dir, exist_ok=True)
    
    df_slide = df[df["slide_ID"] == slide]
    
    # for each ROI in that slide
    for roi in df_slide["ROI_ID"].unique():
        df_roi = df_slide[df_slide["ROI_ID"] == roi]
        out_file = os.path.join(slide_dir, f"ROI_{roi}.csv")
        df_roi.to_csv(out_file, index=False)
        print(f"Wrote {len(df_roi)} rows → {out_file}")


# In[ ]:


import pandas as pd

# load your table (adjust path/reader if it’s an Excel file)
df = pd.read_csv("/Volumes/My Passport/Spatial_Proteomics_data_final/cell_metadata_muspan_combined_rois_all.csv")

# group by broad_type and collect unique cell_types
unique_cell_types = df.groupby("broad_type")["cell_type"].unique()

# display
for btype, ctypes in unique_cell_types.items():
    print(f"{btype}:")
    for ct in ctypes:
        print(f"  - {ct}")


# In[ ]:


import pandas as pd
import os

# I/O paths
in_path = "/Volumes/My Passport/Spatial_Proteomics_data_final/cell_metadata_muspan_combined_rois_all.csv"
out_dir = "/Volumes/My Passport/Spatial_Proteomics_data_final"
os.makedirs(out_dir, exist_ok=True)

# threshold for “low”
THRESH = 4

df = pd.read_csv(in_path)

# 1) count cells per (region, slide, ROI, broad_type)
counts = (
    df
    .groupby(["region","slide_ID","ROI_ID","broad_type"])
    .size()
    .unstack(fill_value=0)   # each broad_type becomes its own column
    .reset_index()
)

# make sure both columns exist even if one type never appears
for b in ["Immune Cells","CAFs"]:
    if b not in counts:
        counts[b] = 0

# pull region‐wide subtype lists for annotation
subtypes = {
    b: sorted(df.loc[df["broad_type"]==b, "cell_type"].dropna().unique())
    for b in ["Immune Cells","CAFs"]
}
sub_str = {b: ", ".join(subtypes[b]) for b in subtypes}

# helper to format Slide_# and ROI_#
slide_fmt = lambda s: f"Slide_{s}"
roi_fmt   = lambda r: f"ROI_{r}"

# iterate regions
for region, group in counts.groupby("region"):
    recs = []
    for b in ["Immune Cells","CAFs"]:
        # find ROIs where this broad_type count is <= THRESH
        low_mask = group[b] <= THRESH
        for _, row in group[low_mask].iterrows():
            recs.append({
                "broad_type": b,
                "cell_type":   sub_str[b],
                "Slide":       slide_fmt(row["slide_ID"]),
                "ROI":         roi_fmt(row["ROI_ID"]),
            })

    if not recs:
        print(f"→ {region}: no low-count ROIs")
        continue

    out_df = pd.DataFrame(recs)
    fname  = f"Exclude_{region.replace(' ','_')}_rois.csv"
    out_df.to_csv(os.path.join(out_dir, fname), index=False)
    print(f"→ {region}: wrote {len(out_df)} rows to {fname}")


# In[ ]:


import pandas as pd
import os
import numpy as np

# I/O paths
in_path = "/Volumes/My Passport/Spatial_Proteomics_data_final/cell_metadata_muspan_combined_rois_all.csv"
out_dir = "/Volumes/My Passport/Spatial_Proteomics_data_final"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(in_path)

broad_types = ["Immune Cells","CAFs"]

# 1) count cells per (region, slide, ROI, broad_type)
counts = (
    df
    .groupby(["region","slide_ID","ROI_ID","broad_type"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# make sure all columns exist even if one type never appears
for b in broad_types:
    if b not in counts:
        counts[b] = 0

# --- CALCULATE 5th PERCENTILE THRESHOLDS ACROSS ALL ROIs ---
THRESHOLDS = {}
for b in broad_types:
    percentile_val = np.percentile(counts[b], 10)
    print(f"10th percentile for {b}: {percentile_val:.0f} cells per ROI")
    THRESHOLDS[b] = percentile_val

# pull region‐wide subtype lists for annotation
subtypes = {
    b: sorted(df.loc[df["broad_type"]==b, "cell_type"].dropna().unique())
    for b in broad_types
}
sub_str = {b: ", ".join(subtypes[b]) for b in subtypes}

# helper to format Slide_# and ROI_#
slide_fmt = lambda s: f"Slide_{s}"
roi_fmt   = lambda r: f"ROI_{r}"

# iterate regions
for region, group in counts.groupby("region"):
    recs = []
    for b in broad_types:
        thresh = THRESHOLDS[b]
        # Exclude ROIs where this broad_type count is < 5th percentile
        low_mask = group[b] < thresh
        for _, row in group[low_mask].iterrows():
            recs.append({
                "broad_type": b,
                "cell_type":   sub_str[b],
                "Slide":       slide_fmt(row["slide_ID"]),
                "ROI":         roi_fmt(row["ROI_ID"]),
            })

    if not recs:
        print(f"→ {region}: no low-count ROIs")
        continue

    out_df = pd.DataFrame(recs)
    out_df = out_df.drop_duplicates(subset=["Slide", "ROI"])
    fname  = f"Exclude_{region.replace(' ','_')}_rois.csv"
    out_df.to_csv(os.path.join(out_dir, fname), index=False)
    print(f"→ {region}: wrote {len(out_df)} rows to {fname}")

# Save thresholds as a CSV for documentation
pd.DataFrame([{"broad_type": b, "10th_percentile_cells": int(THRESHOLDS[b])} for b in broad_types]).to_csv(
    os.path.join(out_dir, "Exclusion_thresholds_10th_percentile.csv"), index=False
)
print(f"Saved 10th percentile thresholds to: {os.path.join(out_dir, 'Exclusion_thresholds_10th_percentile.csv')}")


# In[ ]:


import pandas as pd
import os
import numpy as np

# I/O paths
in_path = "/Volumes/My Passport/Spatial_Proteomics_data_final/cell_metadata_muspan_combined_rois_all.csv"
out_dir = "/Volumes/My Passport/Spatial_Proteomics_data_final"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(in_path)

broad_types = ["Immune Cells", "CAFs"]

# 1) Count cells per (region, slide, ROI, broad_type)
counts = (
    df
    .groupby(["region", "slide_ID", "ROI_ID", "broad_type"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# Ensure all columns exist
for b in broad_types:
    if b not in counts:
        counts[b] = 0
        
# --- ADD THIS LINE: get only Healthy rows ---
healthy_counts = counts[counts["region"] == "Healthy"]

# 2) Calculate median cell count for each broad type in **Healthy** ROIs
MEDIAN_THRESHOLDS = {b: healthy_counts[b].median() for b in broad_types}
for b in broad_types:
    print(f"Median {b} in Healthy region: {MEDIAN_THRESHOLDS[b]:.2f} cells per ROI")

# Save thresholds as a CSV for documentation
pd.DataFrame([
    {"broad_type": b, "Healthy_median_cells": MEDIAN_THRESHOLDS[b]}
    for b in broad_types
]).to_csv(os.path.join(out_dir, "Exclusion_thresholds_Healthy_median.csv"), index=False)
print(f"Saved median thresholds to: {os.path.join(out_dir, 'Exclusion_thresholds_Healthy_median.csv')}")

# 3) Pull region-wide subtype lists for annotation
subtypes = {
    b: sorted(df.loc[df["broad_type"]==b, "cell_type"].dropna().unique())
    for b in broad_types
}
sub_str = {b: ", ".join(subtypes[b]) for b in subtypes}

# Helper to format Slide_# and ROI_#
slide_fmt = lambda s: f"Slide_{s}"
roi_fmt   = lambda r: f"ROI_{r}"

# 4) Iterate regions: exclude by threshold from healthy medians
for region, group in counts.groupby("region"):
    recs = []
    for b in broad_types:
        thresh = MEDIAN_THRESHOLDS[b]
        # Flag ROIs with < median Healthy for this broad type
        low_mask = group[b] < thresh
        for _, row in group[low_mask].iterrows():
            recs.append({
                "broad_type": b,
                "cell_type":   sub_str[b],
                "Slide":       slide_fmt(row["slide_ID"]),
                "ROI":         roi_fmt(row["ROI_ID"]),
            })
    if not recs:
        print(f"→ {region}: no low-count ROIs")
        continue
    out_df = pd.DataFrame(recs)
    out_df = out_df.drop_duplicates(subset=["Slide", "ROI"])
    fname  = f"Exclude_{region.replace(' ','_')}_rois.csv"
    out_df.to_csv(os.path.join(out_dir, fname), index=False)
    print(f"→ {region}: wrote {len(out_df)} rows to {fname}")


# In[ ]:


import os
import pandas as pd

# ─── Inputs ───────────────────────────────────────────────────────────────────
input_path = "/Volumes/My Passport/Spatial_Proteomics_data_final/cell_metadata_with_clusters_9_combat.csv"
output_dir = "/Volumes/My Passport/Spatial_Proteomics_data_final/Region_with_cluster_csvs"

# ─── Prep ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(input_path)
df["slide_ID"] = df["slide_ID"].astype(int)
df["ROI_ID"]   = df["ROI_ID"].astype(int)

# ─── 1) Per‐region “muspan” files ───────────────────────────────────────────────
os.makedirs(output_dir, exist_ok=True)
for region in df["region"].unique():
    sub = df[df["region"] == region].copy().reset_index(drop=True)
    safe = region.replace(" ", "_")
    out = os.path.join(output_dir, f"cell_metadata_muspan_{safe}.csv")
    sub.to_csv(out, index=False)
    print(f"[Region] '{region}': {len(sub)} rows → {out}")

# ─── 2) Per‐ROI files ───────────────────────────────────────────────────────────
for (region, slide, roi), sub in df.groupby(["region","slide_ID","ROI_ID"]):
    # build slide folder
    path = os.path.join(output_dir, region, f"Slide_{slide}")
    os.makedirs(path, exist_ok=True)

    fn = os.path.join(path, f"ROI_{roi}.csv")
    sub.to_csv(fn, index=False)
    print(f"[ROI] {region}/Slide_{slide}/ROI_{roi}.csv")

print("All done!")


# In[ ]:





# In[ ]:




