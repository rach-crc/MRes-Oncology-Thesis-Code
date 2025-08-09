#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Create Spatial_Transcriptomics_data_final_parent manually.
# Manually selected more Peritumour region fovs from those fovs with three region labells (0609).


# In[ ]:


import pandas as pd
# 1. load
infile  = '/Volumes/My Passport/OneDrive_1_01-06-2025/cell_metadata_summary_for_muspan.csv'
outfile = '/Volumes/My Passport/OneDrive_1_01-06-2025/cell_metadata_manual_peritumour_for_muspan.csv'
df = pd.read_csv(infile)
# 2. specify your slide→fov lists
slide_fovs = {
    1:  [140, 159, 201, 151, 133,  73,   5],
    2:  [219, 241, 263, 285, 371, 394, 465, 488, 483, 358, 381, 276, 214, 442, 430],
    4:  [ 23,  77,  79,  80, 100, 123, 142, 199, 240, 282,
         372, 413, 395, 342, 307, 308, 187, 224, 245,  86,  38, 218, 205, 289]
}
# 3. build a boolean mask for rows matching any (slide, fov)
mask = False
for slide, fovs in slide_fovs.items():
    mask |= ( (df['slide_ID'] == slide) &
              (df['fov'].isin(fovs)) )
df_sel = df[mask].copy()
# 4. overwrite region
df_sel['region'] = 'Peritumour'
# 5. write out
df_sel.to_csv(outfile, index=False)
# 6. report
#    count distinct slide–fov pairs in your subset
n_pairs = df_sel[['slide_ID','fov']].drop_duplicates().shape[0]
print(f'Extracted {len(df_sel):,} rows covering {n_pairs} unique slide–fov pairs.')


# In[ ]:


# Combine to get an inital Peritumour (50% all) dataset, change region to Peritumour for all cells.
# cell_metadata_muspan_peritumour_tumour_50.csv: > 50% of cell are from peritumour in the peritumour + tumour region
# cell_metadata_muspan_peritumour_healthy_50.csv: > 50% of cell are from peritumour in the peritumour + healthy region


# In[ ]:


import os
import pandas as pd

# ─── User parameters ─────────────────────────────────────────────────────────
input_path1 = '/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_peritumour_tumour_50.csv'
input_path2 = '/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_peritumour_healthy_50.csv'
output_path = '/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_peritumour_all_50.csv'

# ─── Load both CSVs (preserve their original index) ──────────────────────────
# We tell pandas to treat the first column in each file as the index, rather than
# creating a new RangeIndex. Adjust index_col=0 only if your CSV’s first column
# truly *is* the original index; otherwise omit index_col.
df1 = pd.read_csv(input_path1, index_col=0)
df2 = pd.read_csv(input_path2, index_col=0)

# ─── Force all region labels to "Peritumour" ────────────────────────────
for df in (df1, df2):
    if 'region' in df.columns:
        df['region'] = 'Peritumour'

# ─── Concatenate them (keeping each DataFrame’s original index) ──────────────
combined = pd.concat([df1, df2], ignore_index=False)

# ─── Save the combined file (write out the index as it was) ───────────────────
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# By default, pandas.to_csv(..., index=True) will include the existing index as the first column.
combined.to_csv(output_path, index=True)

print(f"Wrote combined metadata (with forced region) to:\n  {output_path}")


# In[ ]:


# Merge the new peritumour regions with initial peritumour dataset.


# In[ ]:


import pandas as pd
# paths
all50   = '/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_peritumour_all_50.csv'
manual  = '/Volumes/My Passport/OneDrive_1_01-06-2025/cell_metadata_manual_peritumour_for_muspan.csv'
outpath = all50
# load
df50  = pd.read_csv(all50)
dfman = pd.read_csv(manual)
# stack them
df = pd.concat([df50, dfman], ignore_index=True)
# drop duplicates by cell_id (keep the first occurrence)
df = df.drop_duplicates(subset='cell_id', keep='first')
# save
df.to_csv(outpath, index=False)
print(f'Final table: {len(df):,} rows saved to\n{outpath}')


# In[ ]:


# new peritumour 2: cell_metadata_muspan_tumour_peritumour_50.csv minus cell_metadata_muspan_tumour_peritumour_60.csv 


# In[ ]:


import pandas as pd

file50  = '/Volumes/My Passport/OneDrive_1_01-06-2025/cell_metadata_muspan_tumour_peritumour_50.csv'
file60  = '/Volumes/My Passport/OneDrive_1_01-06-2025/cell_metadata_muspan_tumour_peritumour_60.csv'
outpath = '/Volumes/My Passport/OneDrive_1_01-06-2025/cell_metadata_peritumour_60-50tumour.csv'

df50 = pd.read_csv(file50)
df60 = pd.read_csv(file60)

# build sets of (slide, fov)
pairs50 = set(zip(df50['slide_ID'], df50['fov']))
pairs60 = set(zip(df60['slide_ID'], df60['fov']))

# find the extra ones in 50 but not in 60
extra_pairs = pairs50 - pairs60

# subset df50
mask = df50.apply(lambda r: (r['slide_ID'], r['fov']) in extra_pairs, axis=1)
df_extra = df50[mask].copy()
df_extra['region'] = 'Peritumour'

df_extra.to_csv(outpath, index=False)

print(f"Found {len(extra_pairs)} extra slide–fov pairs, extracted {len(df_extra):,} rows.")


# In[ ]:


# Merge the new peritumour regions with initial peritumour dataset again.


# In[ ]:


import pandas as pd
# paths
all50   = '/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_peritumour_all_50.csv'
manual  = '/Volumes/My Passport/OneDrive_1_01-06-2025/cell_metadata_peritumour_60-50tumour.csv'
outpath = all50
# load
df50  = pd.read_csv(all50)
dfman = pd.read_csv(manual)
# stack them
df = pd.concat([df50, dfman], ignore_index=True)
# drop duplicates by cell_id (keep the first occurrence)
df = df.drop_duplicates(subset='cell_id', keep='first')
# save
df.to_csv(outpath, index=False)
print(f'Final table: {len(df):,} rows saved to\n{outpath}')


# In[ ]:


# Get an inital Tumour (> 60%) dataset, change region to Tumour for all cells.
# Tumour_60:cell_metadata_muspan_tumour_peritumour_60.csv: > 60% of cell are from tumour in the peritumour + tumour region


# In[ ]:


import pandas as pd
import os

# –––––– Change this to the exact path of your tumour_peritumour CSV –––––––––
input_path  = "/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_tumour_peritumour_60.csv"

# Overwrite that same file, set output_path = input_path.
output_path = "/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_tumour_peritumour_60_new.csv"

# ─── Load the CSV, keeping its existing index column ───────────────────────────
#
# We assume the first column of the CSV *is* the original index (e.g. cell IDs).
# If that’s not true, remove index_col=0 and pandas will create a new RangeIndex instead.
df = pd.read_csv(input_path, index_col=0)

# ─── Force all values in “region” to be "Tumour" ────────────────────────────────
if "region" in df.columns:
    df["region"] = "Tumour"
else:
    # If there is no 'region' column, create it (filled with "Tumour"):
    df["region"] = "Tumour"

# ─── Save it back out, preserving the index as the first column ─────────────────
# Writing with index=True means that pandas will include the existing index in the CSV.
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=True)

print(f"Wrote file with forced region='Tumour':\n  {output_path}")


# In[ ]:


# Healthy: slide 2 edge fovs removal.


# In[ ]:


import pandas as pd
import os

# 1) Set these paths to wherever your CSVs actually live.
cell_metadata_path = "/Volumes/My Passport/OneDrive_1_01-06-2025/cell_metadata_muspan_healthy.csv"
remove_list_path   = "/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/Slide_2_remove_healthy_fovs.csv"

# This is where the filtered file will be written.
output_path        = "/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_healthy_filtered.csv"

# 2) Read in both CSVs
df_cells = pd.read_csv(cell_metadata_path)
df_rem   = pd.read_csv(remove_list_path)

# 3) Ensure the key columns exist
for df_name, df in [("cell‐metadata", df_cells), ("remove‐list", df_rem)]:
    for col in ("slide_ID", "fov"):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {df_name} DataFrame. "
                             f"Found {df.columns.tolist()} instead.")

# 4) Make sure slide_ID and fov are ints (so merges line up)
df_cells["slide_ID"] = df_cells["slide_ID"].astype(int)
df_cells["fov"]      = df_cells["fov"].astype(int)

df_rem["slide_ID"]   = df_rem["slide_ID"].astype(int)
df_rem["fov"]        = df_rem["fov"].astype(int)

# 5) Build a unique list of (slide_ID, fov) pairs we want to remove
df_rem_pairs = df_rem[["slide_ID", "fov"]].drop_duplicates().reset_index(drop=True)

# 6) Find the intersection of df_cells and df_rem_pairs
df_matched = pd.merge(
    df_cells[["slide_ID", "fov"]],
    df_rem_pairs,
    on=["slide_ID", "fov"],
    how="inner"
).drop_duplicates().reset_index(drop=True)

matched_pairs = list(df_matched.itertuples(index=False, name=None))
# matched_pairs is a list of (slide_ID, fov) that actually appear in df_cells.

# 7) Find which requested pairs were NOT in df_cells
df_missing = pd.merge(
    df_rem_pairs,
    df_matched,
    on=["slide_ID", "fov"],
    how="left",
    indicator=True
)
df_missing = df_missing[df_missing["_merge"] == "left_only"][["slide_ID", "fov"]]
missing_pairs = list(df_missing.itertuples(index=False, name=None))

# 8) Filter out all rows in df_cells whose (slide_ID, fov) is in matched_pairs
df_filtered = pd.merge(
    df_cells,
    df_matched,
    on=["slide_ID", "fov"],
    how="left",
    indicator=True
)
df_filtered = df_filtered[df_filtered["_merge"] == "left_only"].drop(columns=["_merge"])

# 9) Print a summary
print("\n==== SUMMARY OF REMOVAL ====\n")

if matched_pairs:
    print("Removed the following (slide_ID, fov) pairs (they were present in the cell‐metadata):")
    for slide_id, fov in matched_pairs:
        print(f"  • slide_ID = {slide_id},  fov = {fov}")
else:
    print("No (slide_ID, fov) pairs from the removal list were found in cell‐metadata. Nothing was removed.")

print("\n")

if missing_pairs:
    print("The following (slide_ID, fov) pairs were in the removal list but NOT FOUND in cell‐metadata (skipped):")
    for slide_id, fov in missing_pairs:
        print(f"  ◦ slide_ID = {slide_id},  fov = {fov}")
else:
    print("All (slide_ID, fov) pairs in the removal list were found in cell‐metadata.")

print("\n==== END OF SUMMARY ====\n")

# 10) Write out the filtered DataFrame to a new CSV
out_dir = os.path.dirname(output_path)
os.makedirs(out_dir, exist_ok=True)  # make sure the folder exists

df_filtered.to_csv(output_path, index=False)
print(f"Wrote filtered cell‐metadata to:\n  {output_path}")


# In[ ]:


# Combine before modification fovs, cell_metadata_muspan_all-regions.csv


# In[ ]:


import os
import pandas as pd

# ─── Paths to your three source CSVs ───────────────────────────────────────────
path_tumour_peritumour = "/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_tumour_peritumour_60_new.csv"
path_peritumour_all    = "/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_peritumour_all_50.csv"
path_healthy_filtered  = "/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_healthy_filtered.csv"

# ─── Path for the final concatenated CSV ──────────────────────────────────────
output_path = "/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_all-regions.csv"

# ─── 1) Read each CSV, preserving its original index (assuming its first column is the index) ───
df_tumour_peritumour = pd.read_csv(path_tumour_peritumour, index_col=0)
df_peritumour_all    = pd.read_csv(path_peritumour_all,    index_col=0)
df_healthy_filtered  = pd.read_csv(path_healthy_filtered,  index_col=0)

# (If any of these files does NOT actually have its true index as the first column,
# remove `index_col=0`, but then you will lose the “original index” in the merged file.)

# ─── 2) (Optional) Double‐check that all three DataFrames share the same column set ──────────
cols1 = set(df_tumour_peritumour.columns)
cols2 = set(df_peritumour_all.columns)
cols3 = set(df_healthy_filtered.columns)
print("Tumour/Peritumour columns:", cols1)
print("Peritumour_all columns:   ", cols2)
print("Healthy_filtered columns:  ", cols3)
# If anything differs, adjust them here (e.g. rename or add missing columns) so they align.

# ─── 3) Concatenate all three DataFrames, keeping each one’s index ─────────────────────────
combined = pd.concat(
    [df_tumour_peritumour, df_peritumour_all, df_healthy_filtered],
    ignore_index=False
)

# ─── 4) Write out the combined DataFrame, including its index as the first column ──────────
os.makedirs(os.path.dirname(output_path), exist_ok=True)
combined.to_csv(output_path, index=True)

print(f"Wrote merged metadata (all regions) to:\n  {output_path}")


# In[ ]:


# Region modification for specific slides and fovs


# In[ ]:


import os
import pandas as pd

# ─── 1) Define all file‐paths ──────────────────────────────────────────────────

# (a) The merged cell‐metadata that we want to modify:
main_csv_path  = "/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_all-regions.csv"

# (b) The guide: each row is (slide_ID, fov, region) – telling us how to overwrite
guide_csv_path = "/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/slide_fov_region_modification_guide.csv"

# (c) The output file (with updated region).  You can overwrite main_csv_path by
#     setting output_csv_path = main_csv_path, but here we’ll write to a new name:
output_csv_path = "/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_all-regions_modified.csv"


# ─── 2) Load “main” CSV, preserving its original index column ───────────────────
#
# We assume the first column in main_csv_path is the original row‐index (e.g. cell ID).
# If that first column is not actually the index, remove index_col=0, but then you’ll
# lose the “original index” on output.
df_main = pd.read_csv(main_csv_path, index_col=0)

# ─── 3) Load the guide CSV (slide_ID, fov, region) ───────────────────────────────
df_guide = pd.read_csv(guide_csv_path)

# ─── 4) Ensure “slide_ID” and “fov” are the same dtype in both DataFrames ─────────
for name, df in [("main", df_main), ("guide", df_guide)]:
    for col in ("slide_ID", "fov"):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame '{name}'. "
                             f"Found columns: {df.columns.tolist()}")
# Force both to integer (in case they were read as strings or floats)
df_main["slide_ID"] = df_main["slide_ID"].astype(int)
df_main["fov"]      = df_main["fov"].astype(int)

df_guide["slide_ID"] = df_guide["slide_ID"].astype(int)
df_guide["fov"]      = df_guide["fov"].astype(int)

# ─── 5) Build a unique set of guide‐pairs (slide_ID, fov, new_region) ─────────────
df_guide_pairs = df_guide[["slide_ID", "fov", "region"]].drop_duplicates().reset_index(drop=True)

# ─── 6) Find which (slide_ID, fov) from the guide actually exist in df_main ─────────
#
#   (a) Inner‐merge on slide_ID, fov to get only those guide rows that match the main file:
df_matched = pd.merge(
    df_guide_pairs,
    df_main[["slide_ID", "fov"]],
    on=["slide_ID", "fov"],
    how="inner"
).reset_index(drop=True)

# Now df_matched has exactly the guide‐rows that will be applied.  Extract them into a list:
matched_pairs = list(df_matched[["slide_ID", "fov", "region"]].itertuples(index=False, name=None))
# matched_pairs is something like [(1, 298, "Peritumour"), (1, 299, "Peritumour"), (3,  34, "Tumour"), …]

#   (b) Find which guide (slide_ID, fov) were NOT found in df_main:
df_missing = pd.merge(
    df_guide_pairs,
    df_main[["slide_ID", "fov"]],
    on=["slide_ID", "fov"],
    how="left",
    indicator=True
)
df_missing = df_missing[df_missing["_merge"] == "left_only"][["slide_ID", "fov"]]
missing_pairs = list(df_missing.itertuples(index=False, name=None))
# missing_pairs is a list of (slide_ID, fov) that were in the guide but don't exist in df_main.


# ─── 7) Apply the region‐overwrites to df_main ───────────────────────────────────
#
# Easiest way: merge df_main with df_guide_pairs to bring in a “region_new” column,
# then where region_new isn’t null, replace the old region. Finally, drop “region_new”.
#
df_merged = pd.merge(
    df_main.reset_index(),          # bring index back into a column so we don’t lose it
    df_guide_pairs,
    on=["slide_ID", "fov"],
    how="left",                     # keep ALL rows of df_main, attach guide.region where it matches
    suffixes=("", "_new")           # “region” is from df_main, “region_new” from df_guide
)

# At this point, df_merged has columns:
#   [ original_index_col, slide_ID, fov, …, region, …, region_new ]
#
# We now overwrite “region” with “region_new” wherever region_new is not null:
df_merged["region"] = df_merged["region_new"].combine_first(df_merged["region"])

# Drop the extra column “region_new”
df_merged.drop(columns=["region_new"], inplace=True)

# Finally, restore the original index as the DataFrame’s index again:
#    Suppose the original index column was named “cell_id” (or whatever).
#    When we did reset_index(), it became an actual column.  Let’s put it back:
original_index_name = df_main.index.name
df_merged.set_index(original_index_name, inplace=True)

# Now df_merged has exactly the same columns as df_main, but with “region” overwritten
# for any matched (slide_ID, fov).  Rows that did NOT match remain with their old region.


# ─── 8) Print a quick summary of which pairs were applied vs. missing ────────────
print("\n==== REGION‐MODIFICATION SUMMARY ====\n")

# 1) How many pairs were found & updated?
n_matched = len(matched_pairs)
if n_matched:
    print(f"✔ {n_matched} (slide_ID, fov) pairs found in the merged file and updated.")
    # Optionally, show the first few examples:
    print("  Examples:")
    for (slide_id, fov, new_region) in matched_pairs[:5]:
        print(f"    • (slide_ID={slide_id}, fov={fov}) → region='{new_region}'")
    if n_matched > 5:
        print(f"    ...and {n_matched-5} more.\n")
else:
    print("⚠ No (slide_ID, fov) pairs from the guide were found in the merged file. Nothing was updated.\n")

# 2) How many guide‐pairs were missing (not found in the merged file)?
n_missing = len(missing_pairs)
if n_missing:
    print(f"⚠ {n_missing} (slide_ID, fov) pairs were in the guide but NOT FOUND in the merged file.")
    # Optionally, show the first few missing pairs:
    print("  Examples of missing:")
    for (slide_id, fov) in missing_pairs[:5]:
        print(f"    • (slide_ID={slide_id}, fov={fov})")
    if n_missing > 5:
        print(f"    ...and {n_missing-5} more.")
else:
    print("✔ All (slide_ID, fov) pairs in the guide were present in the merged file.")

print("\n==== END OF SUMMARY ====\n")

# ─── 9) Save out the modified DataFrame, keeping the original index ─────────────────
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
df_merged.to_csv(output_csv_path, index=True)

print(f"Wrote modified cell‐metadata (all‐regions) to:\n  {output_csv_path}")


# In[ ]:


# Add pure tumour and peritumour fovs.


# In[ ]:


import pandas as pd

# Paths to your files
path_pure    = "/Volumes/My Passport/OneDrive_1_01-06-2025/cell_metadata_muspan_tumour_pure.csv"
path_modified = "/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_all-regions_modified.csv"

# 1. Read both into DataFrames
df_pure     = pd.read_csv(path_pure)
df_modified = pd.read_csv(path_modified)

# 2. Concatenate them
df_combined = pd.concat([df_modified, df_pure], ignore_index=False)

# 3. (Optional) If you want to drop any exact‐duplicate rows:
df_combined = df_combined.drop_duplicates()

# 4. Save to disk
out_path = "/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_all-regions_modified.csv"
df_combined.to_csv(out_path, index=False)

print(f"Combined file has {len(df_combined)} rows and {df_combined.shape[1]} columns.")
print(f"Written to {out_path}")


# In[ ]:


# Paths to your files
path_pure    = "/Volumes/My Passport/OneDrive_1_01-06-2025/cell_metadata_muspan_peritumour_pure.csv"
path_modified = "/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_all-regions_modified.csv"

# 1. Read both into DataFrames
df_pure     = pd.read_csv(path_pure)
df_modified = pd.read_csv(path_modified)

# 2. Concatenate them
df_combined = pd.concat([df_modified, df_pure], ignore_index=False)

# 3. (Optional) If you want to drop any exact‐duplicate rows:
df_combined = df_combined.drop_duplicates()

# 4. Save to disk
out_path = "/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_all-regions_modified.csv"
df_combined.to_csv(out_path, index=False)

print(f"Combined file has {len(df_combined)} rows and {df_combined.shape[1]} columns.")
print(f"Written to {out_path}")


# In[ ]:


# ─── 1) Load the modified file ───────────────────────────────────────────────────
path = "/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_all-regions_modified.csv"

# We assume the first column is your original index (cell ID, etc.), so we read it in as index_col=0.
df = pd.read_csv(path, index_col=0)

# ─── 2) Determine whether there's already an "FOV" column of the form "s<slide_ID>f<fov>" ──

if "FOV" in df.columns:
    # If there's already a column named FOV (for example "s1f298"), we can count directly off that.
    df["sxfx"] = df["FOV"].astype(str)
else:
    # Otherwise, we build our own "sxfx" by pasting slide_ID + fov.
    # We first check that slide_ID and fov exist:
    for col in ("slide_ID", "fov"):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in the data. Found columns: {df.columns.tolist()}")
    # Now create the combined string column:
    df["sxfx"] = "s" + df["slide_ID"].astype(int).astype(str) + "f" + df["fov"].astype(int).astype(str)

# ─── 3) Group by region and count unique sxfx ────────────────────────────────────
counts = df.groupby("region")["sxfx"].nunique()

# ─── 4) Display the result ───────────────────────────────────────────────────────
print("Number of unique FOVs per region:\n")
print(counts)

# If you want each region printed on its own line:
for region_name, num_fovs in counts.items():
    print(f"  ▸ {region_name}: {num_fovs} unique FOVs")


# In[ ]:


import pandas as pd

# ─── 1) Load the modified file ───────────────────────────────────────────────────
path = "/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_all-regions_modified.csv"
df = pd.read_csv(path, index_col=0)

# ─── 2) Build or copy in the sxfx column ─────────────────────────────────────────
if "FOV" in df.columns:
    df["sxfx"] = df["FOV"].astype(str)
else:
    for col in ("slide_ID", "fov"):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found. Available: {df.columns.tolist()}")
    df["sxfx"] = "s" + df["slide_ID"].astype(int).astype(str) + "f" + df["fov"].astype(int).astype(str)

# ─── 3) Print pre-relabel counts ─────────────────────────────────────────────────
print("Before relabel:")
pre_counts = df.groupby("region")["sxfx"].nunique()
for region_name, num in pre_counts.items():
    print(f"  ▸ {region_name}: {num} unique FOVs")

# ─── 4) Identify Tumour FOVs with >50 Hepatocyte cells ────────────────────────────
# Count hepatocytes per FOV:
hep_counts = df.groupby("sxfx")["broad_type"].apply(lambda x: (x == "Hepatocyte").sum())

# Which of those are currently labelled "Tumour"?
tumour_fovs = set(df.loc[df["region"] == "Tumour", "sxfx"].unique())

# Select the intersection: tumour FOVs with more than 50 hepatocytes
to_relabel = sorted(set(hep_counts[hep_counts > 50].index) & tumour_fovs)

print(f"\nRe-labeling {len(to_relabel)} FOV(s) with > 50 Hepatocytes:")
for f in to_relabel:
    print("  -", f)

# ─── 5) Apply the relabel ────────────────────────────────────────────────────────
df.loc[
    (df["sxfx"].isin(to_relabel)) &
    (df["region"] == "Tumour"),
    "region"
] = "Peritumour"

# ─── 6) Print post-relabel counts ───────────────────────────────────────────────
print("\nAfter relabel:")
post_counts = df.groupby("region")["sxfx"].nunique()
for region_name, num in post_counts.items():
    print(f"  ▸ {region_name}: {num} unique FOVs")

# ─── 7) (Optional) Save back out to a new CSV ────────────────────────────────────
out_path = "/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_all-regions_modified.csv"
df.to_csv(out_path)
print(f"\nUpdated metadata with relabeled regions written to:\n  {out_path}")


# In[ ]:


import pandas as pd

# 1) Load the metadata file
path = "/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_all-regions_modified.csv"
df = pd.read_csv(path, index_col=0)

# 2) Extract and print all unique broad_type values
unique_broad = sorted(df["broad_type"].unique())
print(f"Found {len(unique_broad)} unique broad_type values:")
for bt in unique_broad:
    print(f"  - {bt}")


# In[ ]:


import pandas as pd

# 1) Load your data
path = "/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_all-regions_modified.csv"
df = pd.read_csv(path, index_col=0)

# 2) Rename the old column
df = df.rename(columns={"broad_type": "old_broad_type"})

# 3) Define your mapping for the new broad_type
mapping = {
    "Lymphoid": "Immune Cells",
    "Myeloid":  "Immune Cells",
    "Fibroblast": "CAFs"
}

# 4) Create the new column, defaulting to the old value when not mapped
df["broad_type"] = df["old_broad_type"].map(mapping).fillna(df["old_broad_type"])

# 5) (Optional) Check your new categories
unique_new = sorted(df["broad_type"].unique())
print(f"New broad_type categories ({len(unique_new)}):")
for cat in unique_new:
    print("  -", cat)

# 6) (Optional) Save out to a new CSV
out_path = "/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_all-regions_modified.csv"
df.to_csv(out_path)
print(f"\nSaved updated metadata to:\n  {out_path}")


# In[ ]:


# Check duplication again.
import pandas as pd
# path to your file
path = '/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_all-regions_modified.csv'
# load
df = pd.read_csv(path)
# count total vs unique rows
total   = len(df)
unique  = df.drop_duplicates().shape[0]
dupes   = total - unique
print(f"Total rows:   {total:,}")
print(f"Unique rows:  {unique:,}")
print(f"Duplicated:   {dupes:,} rows")
if dupes > 0:
    # show a few examples of the duplicated rows
    print("\nExample duplicated rows:")
    print(df[df.duplicated(keep=False)].head())


# In[ ]:


# Separation region.csv
import os
import pandas as pd

# ─── 1) Path to the merged‐modified CSV ────────────────────────────────────────
input_path = "/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_all-regions_modified.csv"

# ─── 2) Base folder for the per‐region outputs ───────────────────────────────
output_dir = "/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/"

# ─── 3) Read the merged file, preserving its original index ──────────────────
#     (We assume the first column of the CSV is the index. If it is not, remove index_col=0.)
df_all = pd.read_csv(input_path, index_col=0)

# ─── 4) Verify “region”, “slide_ID”, and “fov” exist ──────────────────────────
for col in ("region", "slide_ID", "fov"):
    if col not in df_all.columns:
        raise KeyError(f"Column '{col}' not found in {input_path}. Found columns: {df_all.columns.tolist()}")

# ─── 5) Coerce slide_ID and fov to int (in case they were read as strings/floats) ──
df_all["slide_ID"] = df_all["slide_ID"].astype(int)
df_all["fov"]      = df_all["fov"].astype(int)

# ─── 6) Loop over each unique region, filter, count unique FOVs, and write CSV ──
unique_regions = df_all["region"].unique()
print(f"Found regions: {list(unique_regions)}\n")

for region_name in unique_regions:
    # (a) Filter to only this region
    df_region = df_all[df_all["region"] == region_name].copy()
    
    # (b) Count how many rows
    n_rows = len(df_region)
    
    # (c) Count unique (slide_ID, fov) pairs
    n_unique_fovs = (
        df_region[["slide_ID", "fov"]]
        .drop_duplicates()
        .shape[0]
    )
    
    # (d) Build output filename
    safe_region = region_name.replace(" ", "_")
    out_path = os.path.join(output_dir, f"cell_metadata_muspan_{safe_region}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # (e) Write the full‐column CSV, including the index
    df_region.to_csv(out_path, index=True)
    
    # (f) Print summary for this region
    print(f"Region '{region_name}':")
    print(f"  • {n_rows} total rows")
    print(f"  • {n_unique_fovs} unique FOVs (slide_ID/fov pairs)")
    print(f"  • Written to:\n      {out_path}\n")

print("Done splitting and counting unique FOVs for each region.")


# In[ ]:


# Create region folders, slide folders, and ROI csvs.


# In[ ]:


# Tumour


# In[ ]:


import os
import pandas as pd

# ─── User parameters ─────────────────────────────────────────────────────────
input_path  = '/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_Tumour.csv'
output_path = '/Volumes/My Passport/Spatial_Transcriptomics_data_final/cell_metadata_muspan_Tumour.csv'

# Set this to whatever integer you want ROI_IDs to start at:
start_id = 1

# ─── Load and process ────────────────────────────────────────────────────────
df = pd.read_csv(input_path)

# 1) Drop the cell_id column (if present)
if 'cell_id' in df.columns:
    df = df.drop(columns=['cell_id'])

# 2) Build the ROI_ID mapping from FOV → integer
unique_fovs = sorted(df['FOV'].unique())
roi_map = {fov: idx for idx, fov in enumerate(unique_fovs, start=start_id)}
df['ROI_ID'] = df['FOV'].map(roi_map)

# ─── Save new file ────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f'Wrote updated metadata (with region="Tumour", without cell_id) to:\n  {output_path}')

# ─── Print the mapping ───────────────────────────────────────────────────────
print("\nFOV → ROI_ID:")
for fov, rid in roi_map.items():
    print(f"  {fov} → {rid}")


# In[ ]:


import os
import pandas as pd

# ─── User parameters ─────────────────────────────────────────────────────────
base_dir   = '/Volumes/My Passport/Spatial_Transcriptomics_data_final'
infile     = os.path.join(base_dir, 'cell_metadata_muspan_Tumour.csv')

# ─── Load the full table ──────────────────────────────────────────────────────
df = pd.read_csv(infile)

# ─── Split & write ────────────────────────────────────────────────────────────
for region, df_reg in df.groupby('region'):
    region_dir = os.path.join(base_dir, region)
    for slide_id, df_slide in df_reg.groupby('slide_ID'):
        slide_dir = os.path.join(region_dir, f'Slide_{slide_id}')
        os.makedirs(slide_dir, exist_ok=True)
        for roi, df_roi in df_slide.groupby('ROI_ID'):
            out_path = os.path.join(slide_dir, f'ROI_{roi}.csv')
            df_roi.to_csv(out_path, index=False)


# In[ ]:


# Peritumour


# In[ ]:


import os
import pandas as pd

# ─── User parameters ─────────────────────────────────────────────────────────
input_path  = '/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_Peritumour.csv'
output_path = '/Volumes/My Passport/Spatial_Transcriptomics_data_final/cell_metadata_muspan_Peritumour.csv'

# Set this to whatever integer you want ROI_IDs to start at:
start_id = 465

# ─── Load and process ────────────────────────────────────────────────────────
df = pd.read_csv(input_path)

# 1) Drop the cell_id column (if present)
if 'cell_id' in df.columns:
    df = df.drop(columns=['cell_id'])

# 2) Build the ROI_ID mapping from FOV → integer
unique_fovs = sorted(df['FOV'].unique())
roi_map = {fov: idx for idx, fov in enumerate(unique_fovs, start=start_id)}
df['ROI_ID'] = df['FOV'].map(roi_map)

# ─── Save new file ────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f'Wrote updated metadata (with region="Peritumour", without cell_id) to:\n  {output_path}')

# ─── Print the mapping ───────────────────────────────────────────────────────
print("\nFOV → ROI_ID:")
for fov, rid in roi_map.items():
    print(f"  {fov} → {rid}")


# In[ ]:


import os
import pandas as pd

# ─── User parameters ─────────────────────────────────────────────────────────
base_dir   = '/Volumes/My Passport/Spatial_Transcriptomics_data_final'
infile     = os.path.join(base_dir, 'cell_metadata_muspan_Peritumour.csv')

# ─── Load the full table ──────────────────────────────────────────────────────
df = pd.read_csv(infile)

# ─── Split & write ────────────────────────────────────────────────────────────
for region, df_reg in df.groupby('region'):
    region_dir = os.path.join(base_dir, region)
    for slide_id, df_slide in df_reg.groupby('slide_ID'):
        slide_dir = os.path.join(region_dir, f'Slide_{slide_id}')
        os.makedirs(slide_dir, exist_ok=True)
        for roi, df_roi in df_slide.groupby('ROI_ID'):
            out_path = os.path.join(slide_dir, f'ROI_{roi}.csv')
            df_roi.to_csv(out_path, index=False)


# In[ ]:


# Heathly 50% randomly chosen.


# In[ ]:


import pandas as pd
import numpy as np
# Configurations
INFILE = '/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_Healthy.csv'
OUTFILE = '/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_Healthy_50percent.csv'
SEED = 42

# Load data
df = pd.read_csv(INFILE)

# Find unique (slide_ID, fov) combinations
fov_groups = df[['slide_ID', 'fov']].drop_duplicates()

# For each slide, select random 50% of FOVs
np.random.seed(SEED)
selected_fovs = (
    fov_groups.groupby('slide_ID', group_keys=False)
    .apply(lambda x: x.sample(max(1, int(np.ceil(len(x)*0.50))), random_state=SEED))
)

# Print stats per slide
selected_counts = selected_fovs['slide_ID'].value_counts().sort_index()
original_counts = fov_groups['slide_ID'].value_counts().sort_index()
for sid in sorted(original_counts.index):
    print(f"Slide {sid}: originally {original_counts[sid]} FOVs, selected {selected_counts.get(sid, 0)} FOVs")

# Filter all cells that are in the selected FOVs
df_selected = df.merge(selected_fovs, on=['slide_ID', 'fov'], how='inner')

# Save result
df_selected.to_csv(OUTFILE, index=False)
print(f"\nSaved {df_selected.shape[0]} cells from {selected_fovs.shape[0]} FOVs to {OUTFILE}")


# In[ ]:


# ─── User parameters ─────────────────────────────────────────────────────────
input_path  = '/Volumes/My Passport/Spatial_Transcriptomics_data_final_parent/cell_metadata_muspan_Healthy_50percent.csv'
output_path = '/Volumes/My Passport/Spatial_Transcriptomics_data_final/cell_metadata_muspan_Healthy.csv'

# Set this to whatever integer you want ROI_IDs to start at:
start_id = 792

# ─── Load and process ────────────────────────────────────────────────────────
df = pd.read_csv(input_path)

# Drop the cell_id column
if 'cell_id' in df.columns:
    df = df.drop(columns=['cell_id'])

# Build the ROI_ID mapping
unique_fovs = sorted(df['FOV'].unique())
roi_map = {fov: idx for idx, fov in enumerate(unique_fovs, start=start_id)}
df['ROI_ID'] = df['FOV'].map(roi_map)

# ─── Save new file ────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f"Wrote updated metadata (without cell_id) with ROI_ID to:\n  {output_path}")

# ─── Print the mapping ───────────────────────────────────────────────────────
print("\nFOV → ROI_ID:")
for fov, rid in roi_map.items():
    print(f"  {fov} → {rid}")


# In[ ]:


import os
import pandas as pd

# ─── User parameters ─────────────────────────────────────────────────────────
base_dir   = '/Volumes/My Passport/Spatial_Transcriptomics_data_final'
infile     = os.path.join(base_dir, 'cell_metadata_muspan_Healthy.csv')

# ─── Load the full table ──────────────────────────────────────────────────────
df = pd.read_csv(infile)

# ─── Split & write ────────────────────────────────────────────────────────────
for region, df_reg in df.groupby('region'):
    region_dir = os.path.join(base_dir, region)
    for slide_id, df_slide in df_reg.groupby('slide_ID'):
        slide_dir = os.path.join(region_dir, f'Slide_{slide_id}')
        os.makedirs(slide_dir, exist_ok=True)
        for roi, df_roi in df_slide.groupby('ROI_ID'):
            out_path = os.path.join(slide_dir, f'ROI_{roi}.csv')
            df_roi.to_csv(out_path, index=False)


# In[ ]:


import os
import pandas as pd

# ─── Paths to your three source files ────────────────────────────────────────
path_peritumour = '/Volumes/My Passport/Spatial_Transcriptomics_data_final/cell_metadata_muspan_Peritumour.csv'
path_tumour     = '/Volumes/My Passport/Spatial_Transcriptomics_data_final/cell_metadata_muspan_Tumour.csv'
path_healthy    = '/Volumes/My Passport/Spatial_Transcriptomics_data_final/cell_metadata_muspan_Healthy.csv'

# ─── Destination folder and filename for the merged output ───────────────────
output_folder = '/Volumes/My Passport/Spatial_Transcriptomics_data_final'
output_filename = 'cell_metadata_muspan_combined_rois_all.csv'
output_path = os.path.join(output_folder, output_filename)

# ─── Load each CSV ───────────────────────────────────────────────────────────
df_peri_all = pd.read_csv(path_peritumour)
df_tumour_peri = pd.read_csv(path_tumour)
df_healthy = pd.read_csv(path_healthy)

# ─── Concatenate into one DataFrame ──────────────────────────────────────────
df_combined = pd.concat([df_peri_all, df_tumour_peri, df_healthy], ignore_index=True)

# ─── Fill all NaNs with 0 ────────────────────────────────────────────────────
df_combined = df_combined.fillna(0)

# ─── Count how many unique ROI_IDs there are ─────────────────────────────────
num_unique_rois = df_combined['ROI_ID'].nunique()
print(f"Total number of unique ROI_IDs across all samples: {num_unique_rois}")

# ─── Save the combined DataFrame to the specified output folder ──────────────
os.makedirs(output_folder, exist_ok=True)
df_combined.to_csv(output_path, index=True)
print(f"Combined dataset written to:\n  {output_path}")


# In[ ]:


import os
import pandas as pd

# ─── User parameters ─────────────────────────────────────────────────────────
base_dir     = '/Volumes/My Passport/Spatial_Transcriptomics_data_final'
region_names = ['Tumour', 'Peritumour', 'Healthy']  # adjust to your folders

# ─── Where to dump your slide-level CSVs ─────────────────────────────────────
slide_level_root = os.path.join(base_dir, 'Slide_level_csvs')

# ─── Loop over regions → slides → ROI files ───────────────────────────────────
for region in region_names:
    region_roi_dir = os.path.join(base_dir, region)
    if not os.path.isdir(region_roi_dir):
        print(f"Skipping {region!r}: folder not found at {region_roi_dir}")
        continue

    # prepare the output directory for this region
    out_region_dir = os.path.join(slide_level_root, region)
    os.makedirs(out_region_dir, exist_ok=True)

    for slide_folder in os.listdir(region_roi_dir):
        slide_dir = os.path.join(region_roi_dir, slide_folder)
        if not os.path.isdir(slide_dir):
            continue

        # collect all ROI_*.csv files
        roi_dfs = []
        for fname in os.listdir(slide_dir):
            if fname.lower().endswith('.csv') and fname.startswith('ROI_'):
                path = os.path.join(slide_dir, fname)
                roi_dfs.append(pd.read_csv(path))

        if not roi_dfs:
            continue

        # stitch and write one slide-level CSV
        slide_df = pd.concat(roi_dfs, ignore_index=True)
        slide_id = slide_folder  # e.g. 'Slide_1'
        out_path = os.path.join(out_region_dir, f'{slide_id}.csv')
        slide_df.to_csv(out_path, index=False)
        print(f'Wrote {len(slide_df)} rows → {out_path}')


# In[ ]:


import pandas as pd
import os
import numpy as np

# I/O paths
in_path = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/cell_metadata_muspan_combined_rois_all.csv"
out_dir = "/Volumes/My Passport/Spatial_Transcriptomics_data_final"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(in_path)

broad_types = ["Immune Cells", "CAFs", "Endothelial"]

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
input_path = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/cell_metadata_with_clusters_10_combat.csv"
output_dir = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/Region_with_cluster_csvs"

# ─── 1) Ensure output directory exists ────────────────────────────────────────
os.makedirs(output_dir, exist_ok=True)

# ─── 2) Read the full file (cluster comes along automatically) ────────────────
df_all = pd.read_csv(input_path, index_col=0)

# ─── 3) Coerce slide_ID and fov to ints ───────────────────────────────────────
df_all["slide_ID"] = df_all["slide_ID"].astype(int)
df_all["fov"]      = df_all["fov"].astype(int)

# ─── 4) Iterate regions ───────────────────────────────────────────────────────
for region_name in df_all["region"].unique():
    # (a) Filter to this region
    df_region = df_all[df_all["region"] == region_name].copy()
    
    # (b) Count rows
    n_rows = len(df_region)
    
    # (c) Count unique FOVs (slide_ID + fov)
    n_unique_fovs = (
        df_region[["slide_ID", "fov"]]
        .drop_duplicates()
        .shape[0]
    )
    
    # (d) Reset index so we don't get an Unnamed:0 column
    df_region.reset_index(drop=True, inplace=True)
    
    # (e) Build filename and write CSV (no index column)
    safe_region = region_name.replace(" ", "_")
    out_path = os.path.join(output_dir, f"cell_metadata_muspan_{safe_region}.csv")
    df_region.to_csv(out_path, index=False)
    
    # (f) Print summary for this region
    print(f"Region '{region_name}':")
    print(f"  • {n_rows} total rows")
    print(f"  • {n_unique_fovs} unique FOVs (slide_ID/fov pairs)")
    print(f"  • Written to:\n      {out_path}\n")

print("Done splitting and counting unique FOVs for each region.")


# In[ ]:


import os
import pandas as pd

# ─── User parameters ─────────────────────────────────────────────────────────
base_dir   = '/Volumes/My Passport/Spatial_Transcriptomics_data_final/Region_with_cluster_csvs'
infile     = os.path.join(base_dir, 'cell_metadata_muspan_Healthy.csv')

# ─── Load the full table ──────────────────────────────────────────────────────
df = pd.read_csv(infile)

# ─── Split & write ────────────────────────────────────────────────────────────
for region, df_reg in df.groupby('region'):
    region_dir = os.path.join(base_dir, region)
    for slide_id, df_slide in df_reg.groupby('slide_ID'):
        slide_dir = os.path.join(region_dir, f'Slide_{slide_id}')
        os.makedirs(slide_dir, exist_ok=True)
        for roi, df_roi in df_slide.groupby('ROI_ID'):
            out_path = os.path.join(slide_dir, f'ROI_{roi}.csv')
            df_roi.to_csv(out_path, index=False)


# In[ ]:


import os
import pandas as pd

# ─── User parameters ─────────────────────────────────────────────────────────
base_dir   = '/Volumes/My Passport/Spatial_Transcriptomics_data_final/Region_with_cluster_csvs'
infile     = os.path.join(base_dir, 'cell_metadata_muspan_Tumour.csv')

# ─── Load the full table ──────────────────────────────────────────────────────
df = pd.read_csv(infile)

# ─── Split & write ────────────────────────────────────────────────────────────
for region, df_reg in df.groupby('region'):
    region_dir = os.path.join(base_dir, region)
    for slide_id, df_slide in df_reg.groupby('slide_ID'):
        slide_dir = os.path.join(region_dir, f'Slide_{slide_id}')
        os.makedirs(slide_dir, exist_ok=True)
        for roi, df_roi in df_slide.groupby('ROI_ID'):
            out_path = os.path.join(slide_dir, f'ROI_{roi}.csv')
            df_roi.to_csv(out_path, index=False)


# In[ ]:


import os
import pandas as pd

# ─── User parameters ─────────────────────────────────────────────────────────
base_dir   = '/Volumes/My Passport/Spatial_Transcriptomics_data_final/Region_with_cluster_csvs'
infile     = os.path.join(base_dir, 'cell_metadata_muspan_Peritumour.csv')

# ─── Load the full table ──────────────────────────────────────────────────────
df = pd.read_csv(infile)

# ─── Split & write ────────────────────────────────────────────────────────────
for region, df_reg in df.groupby('region'):
    region_dir = os.path.join(base_dir, region)
    for slide_id, df_slide in df_reg.groupby('slide_ID'):
        slide_dir = os.path.join(region_dir, f'Slide_{slide_id}')
        os.makedirs(slide_dir, exist_ok=True)
        for roi, df_roi in df_slide.groupby('ROI_ID'):
            out_path = os.path.join(slide_dir, f'ROI_{roi}.csv')
            df_roi.to_csv(out_path, index=False)


# In[ ]:




