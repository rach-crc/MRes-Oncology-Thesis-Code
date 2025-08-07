#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Step one parallel script to process each ROI for PCF calculation.


# In[ ]:


get_ipython().run_cell_magic('writefile', 'process_prote_rois_pcf.py', 'import os\nimport glob\nimport argparse\nimport pandas as pd\nimport numpy as np\nimport muspan as ms\nimport matplotlib.pyplot as plt\nfrom multiprocessing import Pool\n\n\ndef process_roi(args):\n    csv_path, pcf_out_dir, scatter_out_dir = args\n    roi_name = os.path.splitext(os.path.basename(csv_path))[0]\n    df = pd.read_csv(csv_path)\n\n    # assume CSV already has x_um, y_um—no conversion here\n    coords = df[[\'x_um\', \'y_um\']].to_numpy()\n\n    # build the domain\n    domain = ms.domain(roi_name)\n    domain.add_points(coords, \'Cell centres\')\n    domain.add_labels(\'cell_type\', df[\'cell_type\'])\n    domain.estimate_boundary(\n        method=\'alpha shape\',\n        alpha_shape_kwargs={\'alpha\': 95}\n    )\n\n    # scatter plot of original cell types\n    os.makedirs(scatter_out_dir, exist_ok=True)\n    fig, ax = plt.subplots(figsize=(8, 6))\n    ms.visualise.visualise(domain, \'cell_type\', ax=ax, marker_size=5)\n    ax.set_title(f"{roi_name} – cell_type", pad=14)\n    fig.tight_layout(pad=2.5)\n    fig.savefig(os.path.join(scatter_out_dir, f"{roi_name}.png"), dpi=150, bbox_inches=\'tight\')\n    plt.close(fig)\n\n    # prepare original categories and pairs for PCF\n    orig_cats = df[\'cell_type\'].dropna().unique().tolist()\n    axes = [(\'cell_type\', c) for c in orig_cats]\n\n    # include every combination, including self–self and both directions\n    pairs = [((Ai, Ci), (Aj, Cj))\n             for (Ai, Ci) in axes\n             for (Aj, Cj) in axes]\n\n    # find reference R from the first valid pair\n    r_ref = None\n    for (Ai, Ci), (Aj, Cj) in pairs:\n        try:\n            r_ref, _ = ms.spatial_statistics.cross_pair_correlation_function(\n                domain, (Ai, Ci), (Aj, Cj),\n                max_R=250, annulus_step=5, annulus_width=10,\n                exclude_zero=True, boundary_exclude_distance=10,\n                visualise_output=False\n            )\n            break\n        except ValueError:\n            continue\n\n    if r_ref is None:\n        print(f"[{roi_name}] no valid pairs → SKIPPING PCF")\n        return\n\n    # compute PCFs for each original pair\n    pcf = {}\n    for (Ai, Ci), (Aj, Cj) in pairs:\n        key = f"{Ci} ⟷ {Cj}"\n        try:\n            _, g = ms.spatial_statistics.cross_pair_correlation_function(\n                domain, (Ai, Ci), (Aj, Cj),\n                max_R=250, annulus_step=5, annulus_width=10,\n                exclude_zero=True, boundary_exclude_distance=10,\n                visualise_output=False\n            )\n        except ValueError:\n            g = np.full_like(r_ref, np.nan)\n        pcf[key] = g\n\n    # save PCF CSV\n    os.makedirs(pcf_out_dir, exist_ok=True)\n    out_df = pd.DataFrame(pcf, index=r_ref)\n    out_df.index.name = \'R\'\n    out_path = os.path.join(pcf_out_dir, f"{roi_name}_pcf.csv")\n    out_df.to_csv(out_path)\n    print(f"[{roi_name}] PCF → {out_path}")\n\n\ndef process_slide(slide, input_root, pcf_root, scatter_root, n_workers):\n    in_dir = os.path.join(input_root, slide)\n    pcf_out_dir = os.path.join(pcf_root, slide)\n    scatter_out_dir = os.path.join(scatter_root, slide)\n\n    roi_files = sorted(\n        glob.glob(os.path.join(in_dir, "ROI_*.csv")),\n        key=lambda fn: int(os.path.splitext(os.path.basename(fn))[0].split("_")[-1])\n    )\n\n    print(f"--- [{slide}] {len(roi_files)} ROIs (parallel={n_workers}) ---")\n    tasks = [(fp, pcf_out_dir, scatter_out_dir) for fp in roi_files]\n\n    with Pool(processes=min(n_workers, len(tasks))) as pool:\n        pool.map(process_roi, tasks)\n\n\ndef main():\n    parser = argparse.ArgumentParser(\n        description="Compute PCFs and scatter plots for ROIs in parallel within each slide."\n    )\n    parser.add_argument(\n        "-i", "--input-root", required=True,\n        help="Root folder of condition (e.g. .../Spatial_Transcriptomics_data/Tumour)"\n    )\n    parser.add_argument(\n        "-p", "--pcf-root", required=True,\n        help="Where to write PCF CSVs (e.g. .../Spatial_Transcriptomics_data/PCF/Tumour)"\n    )\n    parser.add_argument(\n        "-s", "--scatter-root", required=True,\n        help="Where to write scatter plots (e.g. .../Spatial_Transcriptomics_data/Scattered/Tumour)"\n    )\n    parser.add_argument(\n        "-n", "--processes", type=int, default=6,\n        help="Number of parallel ROI-level processes per slide (default: 6)"\n    )\n    args = parser.parse_args()\n\n    slides = sorted([\n        d for d in os.listdir(args.input_root)\n        if os.path.isdir(os.path.join(args.input_root, d))\n    ])\n\n    for slide in slides:\n        process_slide(\n            slide, args.input_root, args.pcf_root, args.scatter_root, args.processes\n        )\n\n    print("ALL SLIDES DONE")\n\n\nif __name__ == "__main__":\n    main()\n    \n')


# In[ ]:


get_ipython().system('pwd')


# In[ ]:


python process_prote_rois_pcf.py \
  -i "/Volumes/My Passport/Spatial_Proteomics_data_final/Tumour" \
  -p "/Volumes/My Passport/Spatial_Proteomics_data_final/PCF/Tumour" \
  -s "/Volumes/My Passport/Spatial_Proteomics_data_final/Scattered/Tumour"


# In[ ]:


python process_prote_rois_pcf.py \
  -i "/Volumes/My Passport/Spatial_Proteomics_data_final/Peritumour" \
  -p "/Volumes/My Passport/Spatial_Proteomics_data_final/PCF/Peritumour" \
  -s "/Volumes/My Passport/Spatial_Proteomics_data_final/Scattered/Peritumour"


# In[ ]:


python process_prote_rois_pcf.py \
  -i "/Volumes/My Passport/Spatial_Proteomics_data_final/Healthy" \
  -p "/Volumes/My Passport/Spatial_Proteomics_data_final/PCF/Healthy" \
  -s "/Volumes/My Passport/Spatial_Proteomics_data_final/Scattered/Healthy"


# In[ ]:


# Tumour (alpha debug)


# In[ ]:


import os
import glob

def get_rois(root):
    """
    Scan each slide subfolder under `root` and return a dict:
      { slide_name: { set of ROI basenames without “.csv” } }
    """
    rois = {}
    for slide in os.listdir(root):
        slide_dir = os.path.join(root, slide)
        if not os.path.isdir(slide_dir):
            continue
        pattern = os.path.join(slide_dir, "ROI_*.csv")
        files = glob.glob(pattern)
        names = {os.path.splitext(os.path.basename(f))[0] for f in files}
        rois[slide] = names
    return rois

# ── CONFIGURE HERE ──────────────────────────────────────────────────────────────
region = "Tumour"  # ← change this to whichever region you want to check
base = "/Volumes/My Passport/Spatial_Proteomics_data_final"
input_root = os.path.join(base, region)
pcf_root   = os.path.join(base, "PCF", region)
# ────────────────────────────────────────────────────────────────────────────────

orig = get_rois(input_root)
pcf  = get_rois(pcf_root)

missing = []
for slide, roi_set in orig.items():
    pcf_set = pcf.get(slide, set())
    for roi in sorted(roi_set):
        expected = roi + "_pcf"
        if expected not in pcf_set:
            missing.append((slide, roi))

if not missing:
    print("✅ All ROIs have matching PCF files.")
else:
    print("❌ Missing PCF for the following ROIs:")
    for slide, roi in missing:
        print(f"  Slide '{slide}': {roi}.csv  → expected {roi}_pcf.csv")


# In[ ]:


get_ipython().run_cell_magic('writefile', 'process_prote_missing_rois_pcf.py', '"""\nCompute PCFs and scatter plots for ROI 538 across Tumour slides.\n\nUsage:\n    python process_tans_rois_pcf.py \\\n      -i /path/to/Tumour \\\n      -p /path/to/PCF/Tumour \\\n      -s /path/to/Scattered/Tumour \\\n      --alpha 95\n"""\nimport os\nimport glob\nimport argparse\nimport pandas as pd\nimport numpy as np\nimport muspan as ms\nimport matplotlib.pyplot as plt\nfrom multiprocessing import Pool\n\n# Only process ROI index 538\nTARGET_ROI = 538\n\ndef process_roi(csv_path, pcf_out_dir, scatter_out_dir, alpha):\n    roi_name = os.path.splitext(os.path.basename(csv_path))[0]\n    df = pd.read_csv(csv_path)\n\n    # assume CSV already has x_um, y_um\n    coords = df[[\'x_um\', \'y_um\']].to_numpy()\n\n    # build the domain\n    domain = ms.domain(roi_name)\n    domain.add_points(coords, \'Cell centres\')\n    domain.add_labels(\'cell_type\', df[\'cell_type\'])\n    domain.estimate_boundary(\n        method=\'alpha shape\',\n        alpha_shape_kwargs={\'alpha\': alpha}\n    )\n\n    # scatter plot\n    os.makedirs(scatter_out_dir, exist_ok=True)\n    fig, ax = plt.subplots(figsize=(8, 6))\n    ms.visualise.visualise(domain, \'cell_type\', ax=ax, marker_size=5)\n    ax.set_title(f"{roi_name} – cell_type", pad=14)\n    fig.tight_layout(pad=2.5)\n    fig.savefig(os.path.join(scatter_out_dir, f"{roi_name}.png"), dpi=150, bbox_inches=\'tight\')\n    plt.close(fig)\n\n    # prepare PCF pairs\n    orig_cats = df[\'cell_type\'].dropna().unique().tolist()\n    axes = [(\'cell_type\', c) for c in orig_cats]\n    pairs = [(a, b) for a in axes for b in axes]\n\n    # find reference R\n    r_ref = None\n    for (Ai, Ci), (Aj, Cj) in pairs:\n        try:\n            r_ref, _ = ms.spatial_statistics.cross_pair_correlation_function(\n                domain, (Ai, Ci), (Aj, Cj),\n                max_R=250, annulus_step=5, annulus_width=10,\n                exclude_zero=True, boundary_exclude_distance=10,\n                visualise_output=False\n            )\n            break\n        except ValueError:\n            continue\n\n    if r_ref is None:\n        print(f"[{roi_name}] no valid pairs → SKIPPING PCF")\n        return\n\n    # compute and save PCF\n    pcf = {}\n    for (Ai, Ci), (Aj, Cj) in pairs:\n        key = f"{Ci} ⟷ {Cj}"\n        try:\n            _, g = ms.spatial_statistics.cross_pair_correlation_function(\n                domain, (Ai, Ci), (Aj, Cj),\n                max_R=250, annulus_step=5, annulus_width=10,\n                exclude_zero=True, boundary_exclude_distance=10,\n                visualise_output=False\n            )\n        except ValueError:\n            g = np.full_like(r_ref, np.nan)\n        pcf[key] = g\n\n    os.makedirs(pcf_out_dir, exist_ok=True)\n    out_df = pd.DataFrame(pcf, index=r_ref)\n    out_df.index.name = \'R\'\n    out_df.to_csv(os.path.join(pcf_out_dir, f"{roi_name}_pcf.csv"))\n    print(f"[{roi_name}] PCF saved.")\n\n\ndef process_slide(slide, input_root, pcf_root, scatter_root, alpha):\n    in_dir = os.path.join(input_root, slide)\n    pcf_out_dir = os.path.join(pcf_root, slide)\n    scatter_out_dir = os.path.join(scatter_root, slide)\n\n    # find exactly ROI_538.csv\n    roi_path = os.path.join(in_dir, f"ROI_{TARGET_ROI}.csv")\n    if not os.path.isfile(roi_path):\n        print(f"ROI {TARGET_ROI} not found in slide {slide}, skipping.")\n        return\n\n    print(f"Processing slide {slide}, ROI {TARGET_ROI} (alpha={alpha})")\n    process_roi(roi_path, pcf_out_dir, scatter_out_dir, alpha)\n\n\ndef main():\n    parser = argparse.ArgumentParser(\n        description="Compute PCFs and scatter plots for ROI 538 across Tumour slides"\n    )\n    parser.add_argument("-i", "--input-root", required=True,\n                        help="Tumour root folder (contains slide subfolders)")\n    parser.add_argument("-p", "--pcf-root", required=True,\n                        help="Output PCF root for Tumour")\n    parser.add_argument("-s", "--scatter-root", required=True,\n                        help="Output scatter root for Tumour")\n    parser.add_argument("--alpha", type=float, default=95,\n                        help="Alpha for boundary estimation (default: 95)")\n    args = parser.parse_args()\n\n    slides = [d for d in sorted(os.listdir(args.input_root))\n              if os.path.isdir(os.path.join(args.input_root, d))]\n    for slide in slides:\n        process_slide(slide, args.input_root, args.pcf_root,\n                      args.scatter_root, args.alpha)\n    print("Done.")\n\nif __name__ == "__main__":\n    main()\n')


# In[ ]:


python process_prote_missing_rois_pcf.py \
      -i /Volumes/My\ Passport/Spatial_Proteomics_data_final/Tumour \
      -p /Volumes/My\ Passport/Spatial_Proteomics_data_final/PCF/Tumour \
      -s /Volumes/My\ Passport/Spatial_Proteomics_data_final/Scattered/Tumour \
      --alpha 120


# In[ ]:


# Peritumour (alpha debug)


# In[ ]:


import os
import glob

def get_rois(root):
    """
    Scan each slide subfolder under `root` and return a dict:
      { slide_name: { set of ROI basenames without “.csv” } }
    """
    rois = {}
    for slide in os.listdir(root):
        slide_dir = os.path.join(root, slide)
        if not os.path.isdir(slide_dir):
            continue
        pattern = os.path.join(slide_dir, "ROI_*.csv")
        files = glob.glob(pattern)
        names = {os.path.splitext(os.path.basename(f))[0] for f in files}
        rois[slide] = names
    return rois

# ── CONFIGURE HERE ──────────────────────────────────────────────────────────────
region = "Peritumour"  # ← change this to whichever region you want to check
base = "/Volumes/My Passport/Spatial_Proteomics_data_final"
input_root = os.path.join(base, region)
pcf_root   = os.path.join(base, "PCF", region)
# ────────────────────────────────────────────────────────────────────────────────

orig = get_rois(input_root)
pcf  = get_rois(pcf_root)

missing = []
for slide, roi_set in orig.items():
    pcf_set = pcf.get(slide, set())
    for roi in sorted(roi_set):
        expected = roi + "_pcf"
        if expected not in pcf_set:
            missing.append((slide, roi))

if not missing:
    print("✅ All ROIs have matching PCF files.")
else:
    print("❌ Missing PCF for the following ROIs:")
    for slide, roi in missing:
        print(f"  Slide '{slide}': {roi}.csv  → expected {roi}_pcf.csv")


# In[ ]:


get_ipython().run_cell_magic('writefile', 'process_prote_missing_rois_pcf.py', '"""\nCompute PCFs and scatter plots for ROI 811 across Tumour slides.\n\nUsage:\n    python process_tans_rois_pcf.py \\\n      -i /path/to/Tumour \\\n      -p /path/to/PCF/Tumour \\\n      -s /path/to/Scattered/Tumour \\\n      --alpha 95\n"""\nimport os\nimport glob\nimport argparse\nimport pandas as pd\nimport numpy as np\nimport muspan as ms\nimport matplotlib.pyplot as plt\nfrom multiprocessing import Pool\n\n# Only process ROI index 811\nTARGET_ROI = 811\n\ndef process_roi(csv_path, pcf_out_dir, scatter_out_dir, alpha):\n    roi_name = os.path.splitext(os.path.basename(csv_path))[0]\n    df = pd.read_csv(csv_path)\n\n    # assume CSV already has x_um, y_um\n    coords = df[[\'x_um\', \'y_um\']].to_numpy()\n\n    # build the domain\n    domain = ms.domain(roi_name)\n    domain.add_points(coords, \'Cell centres\')\n    domain.add_labels(\'cell_type\', df[\'cell_type\'])\n    domain.estimate_boundary(\n        method=\'alpha shape\',\n        alpha_shape_kwargs={\'alpha\': alpha}\n    )\n\n    # scatter plot\n    os.makedirs(scatter_out_dir, exist_ok=True)\n    fig, ax = plt.subplots(figsize=(8, 6))\n    ms.visualise.visualise(domain, \'cell_type\', ax=ax, marker_size=5)\n    ax.set_title(f"{roi_name} – cell_type", pad=14)\n    fig.tight_layout(pad=2.5)\n    fig.savefig(os.path.join(scatter_out_dir, f"{roi_name}.png"), dpi=150, bbox_inches=\'tight\')\n    plt.close(fig)\n\n    # prepare PCF pairs\n    orig_cats = df[\'cell_type\'].dropna().unique().tolist()\n    axes = [(\'cell_type\', c) for c in orig_cats]\n    pairs = [(a, b) for a in axes for b in axes]\n\n    # find reference R\n    r_ref = None\n    for (Ai, Ci), (Aj, Cj) in pairs:\n        try:\n            r_ref, _ = ms.spatial_statistics.cross_pair_correlation_function(\n                domain, (Ai, Ci), (Aj, Cj),\n                max_R=250, annulus_step=5, annulus_width=10,\n                exclude_zero=True, boundary_exclude_distance=10,\n                visualise_output=False\n            )\n            break\n        except ValueError:\n            continue\n\n    if r_ref is None:\n        print(f"[{roi_name}] no valid pairs → SKIPPING PCF")\n        return\n\n    # compute and save PCF\n    pcf = {}\n    for (Ai, Ci), (Aj, Cj) in pairs:\n        key = f"{Ci} ⟷ {Cj}"\n        try:\n            _, g = ms.spatial_statistics.cross_pair_correlation_function(\n                domain, (Ai, Ci), (Aj, Cj),\n                max_R=250, annulus_step=5, annulus_width=10,\n                exclude_zero=True, boundary_exclude_distance=10,\n                visualise_output=False\n            )\n        except ValueError:\n            g = np.full_like(r_ref, np.nan)\n        pcf[key] = g\n\n    os.makedirs(pcf_out_dir, exist_ok=True)\n    out_df = pd.DataFrame(pcf, index=r_ref)\n    out_df.index.name = \'R\'\n    out_df.to_csv(os.path.join(pcf_out_dir, f"{roi_name}_pcf.csv"))\n    print(f"[{roi_name}] PCF saved.")\n\n\ndef process_slide(slide, input_root, pcf_root, scatter_root, alpha):\n    in_dir = os.path.join(input_root, slide)\n    pcf_out_dir = os.path.join(pcf_root, slide)\n    scatter_out_dir = os.path.join(scatter_root, slide)\n\n    # find exactly ROI_538.csv\n    roi_path = os.path.join(in_dir, f"ROI_{TARGET_ROI}.csv")\n    if not os.path.isfile(roi_path):\n        print(f"ROI {TARGET_ROI} not found in slide {slide}, skipping.")\n        return\n\n    print(f"Processing slide {slide}, ROI {TARGET_ROI} (alpha={alpha})")\n    process_roi(roi_path, pcf_out_dir, scatter_out_dir, alpha)\n\n\ndef main():\n    parser = argparse.ArgumentParser(\n        description="Compute PCFs and scatter plots for ROI 811 across Peritumour slides"\n    )\n    parser.add_argument("-i", "--input-root", required=True,\n                        help="Tumour root folder (contains slide subfolders)")\n    parser.add_argument("-p", "--pcf-root", required=True,\n                        help="Output PCF root for Tumour")\n    parser.add_argument("-s", "--scatter-root", required=True,\n                        help="Output scatter root for Tumour")\n    parser.add_argument("--alpha", type=float, default=95,\n                        help="Alpha for boundary estimation (default: 95)")\n    args = parser.parse_args()\n\n    slides = [d for d in sorted(os.listdir(args.input_root))\n              if os.path.isdir(os.path.join(args.input_root, d))]\n    for slide in slides:\n        process_slide(slide, args.input_root, args.pcf_root,\n                      args.scatter_root, args.alpha)\n    print("Done.")\n\nif __name__ == "__main__":\n    main()\n    \n')


# In[ ]:


python process_prote_missing_rois_pcf.py \
      -i /Volumes/My\ Passport/Spatial_Proteomics_data_final/Peritumour \
      -p /Volumes/My\ Passport/Spatial_Proteomics_data_final/PCF/Peritumour \
      -s /Volumes/My\ Passport/Spatial_Proteomics_data_final/Scattered/Peritumour \
      --alpha 120


# In[ ]:


# Healthy (alpha debug)


# In[ ]:


import os
import glob

def get_rois(root):
    """
    Scan each slide subfolder under `root` and return a dict:
      { slide_name: { set of ROI basenames without “.csv” } }
    """
    rois = {}
    for slide in os.listdir(root):
        slide_dir = os.path.join(root, slide)
        if not os.path.isdir(slide_dir):
            continue
        pattern = os.path.join(slide_dir, "ROI_*.csv")
        files = glob.glob(pattern)
        names = {os.path.splitext(os.path.basename(f))[0] for f in files}
        rois[slide] = names
    return rois

# ── CONFIGURE HERE ──────────────────────────────────────────────────────────────
region = "Healthy"  # ← change this to whichever region you want to check
base = "/Volumes/My Passport/Spatial_Proteomics_data_final"
input_root = os.path.join(base, region)
pcf_root   = os.path.join(base, "PCF", region)
# ────────────────────────────────────────────────────────────────────────────────

orig = get_rois(input_root)
pcf  = get_rois(pcf_root)

missing = []
for slide, roi_set in orig.items():
    pcf_set = pcf.get(slide, set())
    for roi in sorted(roi_set):
        expected = roi + "_pcf"
        if expected not in pcf_set:
            missing.append((slide, roi))

if not missing:
    print("✅ All ROIs have matching PCF files.")
else:
    print("❌ Missing PCF for the following ROIs:")
    for slide, roi in missing:
        print(f"  Slide '{slide}': {roi}.csv  → expected {roi}_pcf.csv")


# In[ ]:


get_ipython().run_cell_magic('writefile', 'process_missing_rois_pcf_combined.py', '\n"""\nCompute PCFs and scatter plots for specific missing ROIs across slides.\n\nUsage:\n    python process_missing_rois_pcf.py \\\n      -i /path/to/input_root \\\n      -p /path/to/pcf_root \\\n      -s /path/to/scatter_root \\\n      --alpha 120\n"""\nimport os\nimport argparse\nimport pandas as pd\nimport numpy as np\nimport muspan as ms\nimport matplotlib.pyplot as plt\n\n# Specify only the slides and ROIs that need processing\nMISSING_ROIS = {\n    \'Slide_4\': [1019],\n    \'Slide_8\': [1289]\n}\n\ndef process_roi(csv_path, pcf_out_dir, scatter_out_dir, alpha):\n    roi_name = os.path.splitext(os.path.basename(csv_path))[0]\n    df = pd.read_csv(csv_path)\n\n    coords = df[[\'x_um\', \'y_um\']].to_numpy()\n\n    domain = ms.domain(roi_name)\n    domain.add_points(coords, \'Cell centres\')\n    domain.add_labels(\'cell_type\', df[\'cell_type\'])\n    domain.estimate_boundary(\n        method=\'alpha shape\',\n        alpha_shape_kwargs={\'alpha\': alpha}\n    )\n\n    # Scatter plot\n    os.makedirs(scatter_out_dir, exist_ok=True)\n    fig, ax = plt.subplots(figsize=(8, 6))\n    ms.visualise.visualise(domain, \'cell_type\', ax=ax, marker_size=5)\n    ax.set_title(f"{roi_name} – cell_type", pad=14)\n    fig.tight_layout(pad=2.5)\n    fig.savefig(os.path.join(scatter_out_dir, f"{roi_name}.png"), dpi=150, bbox_inches=\'tight\')\n    plt.close(fig)\n\n    # Prepare PCF computation\n    orig_cats = df[\'cell_type\'].dropna().unique().tolist()\n    axes = [(\'cell_type\', c) for c in orig_cats]\n    pairs = [(a, b) for a in axes for b in axes]\n\n    r_ref = None\n    for (Ai, Ci), (Aj, Cj) in pairs:\n        try:\n            r_ref, _ = ms.spatial_statistics.cross_pair_correlation_function(\n                domain, (Ai, Ci), (Aj, Cj),\n                max_R=250, annulus_step=5, annulus_width=10,\n                exclude_zero=True, boundary_exclude_distance=10,\n                visualise_output=False\n            )\n            break\n        except ValueError:\n            continue\n\n    if r_ref is None:\n        print(f"[{roi_name}] no valid pairs → SKIPPING PCF")\n        return\n\n    pcf = {}\n    for (Ai, Ci), (Aj, Cj) in pairs:\n        key = f"{Ci} ⟷ {Cj}"\n        try:\n            _, g = ms.spatial_statistics.cross_pair_correlation_function(\n                domain, (Ai, Ci), (Aj, Cj),\n                max_R=250, annulus_step=5, annulus_width=10,\n                exclude_zero=True, boundary_exclude_distance=10,\n                visualise_output=False\n            )\n        except ValueError:\n            g = np.full_like(r_ref, np.nan)\n        pcf[key] = g\n\n    # Save PCF\n    os.makedirs(pcf_out_dir, exist_ok=True)\n    out_df = pd.DataFrame(pcf, index=r_ref)\n    out_df.index.name = \'R\'\n    out_df.to_csv(os.path.join(pcf_out_dir, f"{roi_name}_pcf.csv"))\n    print(f"[{roi_name}] PCF saved.")\n\n\ndef main():\n    parser = argparse.ArgumentParser(\n        description="Compute PCFs and scatter plots for specified missing ROIs"\n    )\n    parser.add_argument("-i", "--input-root", required=True,\n                        help="Root folder containing slide subfolders")\n    parser.add_argument("-p", "--pcf-root", required=True,\n                        help="Root folder where PCF CSVs will be saved")\n    parser.add_argument("-s", "--scatter-root", required=True,\n                        help="Root folder where scatter plots will be saved")\n    parser.add_argument("--alpha", type=float, default=120,\n                        help="Alpha for boundary estimation (default: 120)")\n    args = parser.parse_args()\n\n    for slide, rois in MISSING_ROIS.items():\n        for roi_id in rois:\n            roi_name = f"ROI_{roi_id}"\n            csv_path = os.path.join(args.input_root, slide, f"{roi_name}.csv")\n            if not os.path.isfile(csv_path):\n                print(f"{roi_name}.csv not found in {slide}, skipping.")\n                continue\n\n            pcf_out_dir = os.path.join(args.pcf_root, slide)\n            pcf_file = os.path.join(pcf_out_dir, f"{roi_name}_pcf.csv")\n            if os.path.isfile(pcf_file):\n                print(f"[{slide}] {roi_name}_pcf.csv already exists, skipping.")\n                continue\n\n            scatter_out_dir = os.path.join(args.scatter_root, slide)\n            print(f"[{slide}] Processing {roi_name} (alpha={args.alpha})")\n            process_roi(csv_path, pcf_out_dir, scatter_out_dir, args.alpha)\n\n    print("Done.")\n\n\nif __name__ == "__main__":\n    main()\n')


# In[ ]:


python process_missing_rois_pcf_combined.py \
  -i "/Volumes/My Passport/Spatial_Proteomics_data_final/Healthy" \
  -p "/Volumes/My Passport/Spatial_Proteomics_data_final/PCF/Healthy" \
  -s "/Volumes/My Passport/Spatial_Proteomics_data_final/Scattered/Healthy" \
  --alpha 120


# In[ ]:


# 2.3.1 Averaged cross-PCF across all ROIs in a Region (without ROI filtration)


# In[ ]:


# Tumour


# In[ ]:


import os
import glob
import pandas as pd
import numpy as np
from collections import Counter

# ── USER SETTINGS ──────────────────────────────────────────
region       = "Tumour"
input_root   = "/Volumes/My Passport/Spatial_Proteomics_data_final/PCF"
output_root  = "/Volumes/My Passport/Spatial_Proteomics_data_final/Averaged_ROIs_Per_Region/PCF"
# ────────────────────────────────────────────────────────────

def average_and_sd_over_all_rois(region_dir, out_csv, count_csv, fill_missing=0.0):
    """
    Compute region-level PCF stats and report ROI usage counts per cell-pair.

    - region_dir:  folder containing Slide_*/ROI_*_pcf.csv files
    - out_csv:     path to write the matrix of R, {pair}_mean, {pair}_std
    - count_csv:   path to write summary of {pair}, num_rois
    - fill_missing: value to use for ROIs that lack a given pair column.
                    e.g. 0.0 to include all ROIs, np.nan to ignore in mean.
    """
    roi_files = glob.glob(os.path.join(region_dir, "Slide_*", "ROI_*_pcf.csv"))
    if not roi_files:
        raise RuntimeError(f"No ROI PCF CSVs found in {region_dir!r}")

    # read all ROIs & track pair presence
    rois = []
    pair_counts = Counter()
    for fn in sorted(roi_files):
        df = pd.read_csv(fn)
        rois.append(df)
        for col in df.columns.drop('R'):
            pair_counts[col] += 1

    valid_pairs = sorted(pair_counts.keys())

    # assume all ROIs share the same R vector; if not, merge on R instead
    r_vals = rois[0]['R'].values
    out_df = pd.DataFrame({'R': r_vals})

    n_rois = len(rois)

    for pair in valid_pairs:
        # collect values, inserting fill_missing when pair absent
        arrs = []
        for df in rois:
            if pair in df.columns:
                arrs.append(df[pair].values)
            else:
                arrs.append(np.full_like(r_vals, fill_missing, dtype=float))
        mat = np.vstack(arrs)

        if np.isnan(fill_missing):
            # exclude NaNs in mean/std (classic nanmean/nanstd)
            mean = np.nanmean(mat, axis=0)
            std  = np.nanstd(mat, axis=0, ddof=0)
        else:
            # include ALL ROIs: denominator is n_rois
            mean = mat.mean(axis=0)            # since we filled with a numeric value
            std  = mat.std(axis=0, ddof=0)

        out_df[f"{pair}_mean"] = mean
        out_df[f"{pair}_std"]  = std

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"→ Wrote PCF mean/std matrix to {out_csv!r}")

    summary = pd.DataFrame({
        'pair': valid_pairs,
        'num_rois': [pair_counts[p] for p in valid_pairs],
        'total_rois': n_rois
    })
    summary.to_csv(count_csv, index=False)
    print(f"→ Wrote pair usage counts to {count_csv!r}")


if __name__ == '__main__':
    region_dir = os.path.join(input_root, region)
    out_dir    = os.path.join(output_root, region)
    os.makedirs(out_dir, exist_ok=True)

    stats_csv  = os.path.join(out_dir, f"batch_{region}_pcf_stats.csv")
    counts_csv = os.path.join(out_dir, f"batch_{region}_pair_counts.csv")

    print(f"\nProcessing region: {region}")
    average_and_sd_over_all_rois(region_dir, stats_csv, counts_csv, fill_missing=0.0)


# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── USER SETTINGS ──────────────────────────────────────────
region         = "Tumour"
stats_csv      = "/Volumes/My Passport/Spatial_Proteomics_data_final/Averaged_ROIs_Per_Region/PCF/Tumour/batch_Tumour_pcf_stats.csv"
counts_csv     = "/Volumes/My Passport/Spatial_Proteomics_data_final/Averaged_ROIs_Per_Region/PCF/Tumour/batch_Tumour_pair_counts.csv"
save_plots_to  = os.path.join(os.path.dirname(stats_csv), "Individual_Pair_Plots")
os.makedirs(save_plots_to, exist_ok=True)

# color map for plotting
color_map = {"Tumour": "orange"}

# load stats and counts
df_stats = pd.read_csv(stats_csv)
df_counts = pd.read_csv(counts_csv).set_index('pair')

# identify all pairs
pair_cols = [col[:-5] for col in df_stats.columns if col.endswith('_mean')]

for pair in sorted(pair_cols):
    # extract R and metrics
    r = df_stats['R'].values
    m = df_stats[f"{pair}_mean"].values
    s = df_stats[f"{pair}_std"].values
    n = df_counts.loc[pair, 'num_rois'] if pair in df_counts.index else np.nan
    # compute 95% CI
    ci = 1.96 * s / np.sqrt(n)

    # plot
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.plot(r, m, label=region, lw=1.5, color=color_map.get(region, 'black'))
    ax.fill_between(r, m-ci, m+ci, alpha=0.2, color=color_map.get(region, 'black'))

    # self-pair highlight
    if '⟷' in pair and pair.split('⟷')[0].strip() == pair.split('⟷')[1].strip():
        ax.set_facecolor('#fff6f6')

    ax.axhline(1, linestyle='--', lw=0.8, color='gray')
    ax.set_ylim(0, None)
    ax.set_xlabel('R (μm)')
    ax.set_ylabel(f"g(r): {pair}", labelpad=5, fontsize=9)
    ax.set_title(pair)
    ax.legend(title='Region')
    fig.tight_layout()

    # save
    out_fn = os.path.join(save_plots_to, f"{pair.replace(' ', '_')}.png")
    fig.savefig(out_fn, dpi=250)
    plt.close(fig)

print(f"✔️ Individual pair plots written to {save_plots_to}")


# In[ ]:


# Peritumour


# In[ ]:


import os
import glob
import pandas as pd
import numpy as np
from collections import Counter

# ── USER SETTINGS ──────────────────────────────────────────
region       = "Peritumour"
input_root   = "/Volumes/My Passport/Spatial_Proteomics_data_final/PCF"
output_root  = "/Volumes/My Passport/Spatial_Proteomics_data_final/Averaged_ROIs_Per_Region/PCF"
# ────────────────────────────────────────────────────────────

def average_and_sd_over_all_rois(region_dir, out_csv, count_csv, fill_missing=0.0):
    """
    Compute region-level PCF stats and report ROI usage counts per cell-pair.

    - region_dir:  folder containing Slide_*/ROI_*_pcf.csv files
    - out_csv:     path to write the matrix of R, {pair}_mean, {pair}_std
    - count_csv:   path to write summary of {pair}, num_rois
    - fill_missing: value to use for ROIs that lack a given pair column.
                    e.g. 0.0 to include all ROIs, np.nan to ignore in mean.
    """
    roi_files = glob.glob(os.path.join(region_dir, "Slide_*", "ROI_*_pcf.csv"))
    if not roi_files:
        raise RuntimeError(f"No ROI PCF CSVs found in {region_dir!r}")

    # read all ROIs & track pair presence
    rois = []
    pair_counts = Counter()
    for fn in sorted(roi_files):
        df = pd.read_csv(fn)
        rois.append(df)
        for col in df.columns.drop('R'):
            pair_counts[col] += 1

    valid_pairs = sorted(pair_counts.keys())

    # assume all ROIs share the same R vector; if not, merge on R instead
    r_vals = rois[0]['R'].values
    out_df = pd.DataFrame({'R': r_vals})

    n_rois = len(rois)

    for pair in valid_pairs:
        # collect values, inserting fill_missing when pair absent
        arrs = []
        for df in rois:
            if pair in df.columns:
                arrs.append(df[pair].values)
            else:
                arrs.append(np.full_like(r_vals, fill_missing, dtype=float))
        mat = np.vstack(arrs)

        if np.isnan(fill_missing):
            # exclude NaNs in mean/std (classic nanmean/nanstd)
            mean = np.nanmean(mat, axis=0)
            std  = np.nanstd(mat, axis=0, ddof=0)
        else:
            # include ALL ROIs: denominator is n_rois
            mean = mat.mean(axis=0)            # since we filled with a numeric value
            std  = mat.std(axis=0, ddof=0)

        out_df[f"{pair}_mean"] = mean
        out_df[f"{pair}_std"]  = std

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"→ Wrote PCF mean/std matrix to {out_csv!r}")

    summary = pd.DataFrame({
        'pair': valid_pairs,
        'num_rois': [pair_counts[p] for p in valid_pairs],
        'total_rois': n_rois
    })
    summary.to_csv(count_csv, index=False)
    print(f"→ Wrote pair usage counts to {count_csv!r}")


if __name__ == '__main__':
    region_dir = os.path.join(input_root, region)
    out_dir    = os.path.join(output_root, region)
    os.makedirs(out_dir, exist_ok=True)

    stats_csv  = os.path.join(out_dir, f"batch_{region}_pcf_stats.csv")
    counts_csv = os.path.join(out_dir, f"batch_{region}_pair_counts.csv")

    print(f"\nProcessing region: {region}")
    average_and_sd_over_all_rois(region_dir, stats_csv, counts_csv, fill_missing=0.0)


# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── USER SETTINGS ──────────────────────────────────────────
region         = "Peritumour"
stats_csv      = "/Volumes/My Passport/Spatial_Proteomics_data_final/Averaged_ROIs_Per_Region/PCF/Peritumour/batch_Peritumour_pcf_stats.csv"
counts_csv     = "/Volumes/My Passport/Spatial_Proteomics_data_final/Averaged_ROIs_Per_Region/PCF/Peritumour/batch_Peritumour_pair_counts.csv"
save_plots_to  = os.path.join(os.path.dirname(stats_csv), "Individual_Pair_Plots")
os.makedirs(save_plots_to, exist_ok=True)

# color map for plotting
color_map = {"Peritumour": "lightblue"}

# load stats and counts
df_stats = pd.read_csv(stats_csv)
df_counts = pd.read_csv(counts_csv).set_index('pair')

# identify all pairs
pair_cols = [col[:-5] for col in df_stats.columns if col.endswith('_mean')]

for pair in sorted(pair_cols):
    # extract R and metrics
    r = df_stats['R'].values
    m = df_stats[f"{pair}_mean"].values
    s = df_stats[f"{pair}_std"].values
    n = df_counts.loc[pair, 'num_rois'] if pair in df_counts.index else np.nan
    # compute 95% CI
    ci = 1.96 * s / np.sqrt(n)

    # plot
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.plot(r, m, label=region, lw=1.5, color=color_map.get(region, 'black'))
    ax.fill_between(r, m-ci, m+ci, alpha=0.2, color=color_map.get(region, 'black'))

    # self-pair highlight
    if '⟷' in pair and pair.split('⟷')[0].strip() == pair.split('⟷')[1].strip():
        ax.set_facecolor('#fff6f6')

    ax.axhline(1, linestyle='--', lw=0.8, color='gray')
    ax.set_ylim(0, None)
    ax.set_xlabel('R (μm)')
    ax.set_ylabel(f"g(r): {pair}", labelpad=5, fontsize=9)
    ax.set_title(pair)
    ax.legend(title='Region')
    fig.tight_layout()

    # save
    out_fn = os.path.join(save_plots_to, f"{pair.replace(' ', '_')}.png")
    fig.savefig(out_fn, dpi=250)
    plt.close(fig)

print(f"✔️ Individual pair plots written to {save_plots_to}")


# In[ ]:


# Healthy


# In[ ]:


import os
import glob
import pandas as pd
import numpy as np
from collections import Counter

# ── USER SETTINGS ──────────────────────────────────────────
region       = "Healthy"
input_root   = "/Volumes/My Passport/Spatial_Proteomics_data_final/PCF"
output_root  = "/Volumes/My Passport/Spatial_Proteomics_data_final/Averaged_ROIs_Per_Region/PCF"
# ────────────────────────────────────────────────────────────

def average_and_sd_over_all_rois(region_dir, out_csv, count_csv, fill_missing=0.0):
    """
    Compute region-level PCF stats and report ROI usage counts per cell-pair.

    - region_dir:  folder containing Slide_*/ROI_*_pcf.csv files
    - out_csv:     path to write the matrix of R, {pair}_mean, {pair}_std
    - count_csv:   path to write summary of {pair}, num_rois
    - fill_missing: value to use for ROIs that lack a given pair column.
                    e.g. 0.0 to include all ROIs, np.nan to ignore in mean.
    """
    roi_files = glob.glob(os.path.join(region_dir, "Slide_*", "ROI_*_pcf.csv"))
    if not roi_files:
        raise RuntimeError(f"No ROI PCF CSVs found in {region_dir!r}")

    # read all ROIs & track pair presence
    rois = []
    pair_counts = Counter()
    for fn in sorted(roi_files):
        df = pd.read_csv(fn)
        rois.append(df)
        for col in df.columns.drop('R'):
            pair_counts[col] += 1

    valid_pairs = sorted(pair_counts.keys())

    # assume all ROIs share the same R vector; if not, merge on R instead
    r_vals = rois[0]['R'].values
    out_df = pd.DataFrame({'R': r_vals})

    n_rois = len(rois)

    for pair in valid_pairs:
        # collect values, inserting fill_missing when pair absent
        arrs = []
        for df in rois:
            if pair in df.columns:
                arrs.append(df[pair].values)
            else:
                arrs.append(np.full_like(r_vals, fill_missing, dtype=float))
        mat = np.vstack(arrs)

        if np.isnan(fill_missing):
            # exclude NaNs in mean/std (classic nanmean/nanstd)
            mean = np.nanmean(mat, axis=0)
            std  = np.nanstd(mat, axis=0, ddof=0)
        else:
            # include ALL ROIs: denominator is n_rois
            mean = mat.mean(axis=0)            # since we filled with a numeric value
            std  = mat.std(axis=0, ddof=0)

        out_df[f"{pair}_mean"] = mean
        out_df[f"{pair}_std"]  = std

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"→ Wrote PCF mean/std matrix to {out_csv!r}")

    summary = pd.DataFrame({
        'pair': valid_pairs,
        'num_rois': [pair_counts[p] for p in valid_pairs],
        'total_rois': n_rois
    })
    summary.to_csv(count_csv, index=False)
    print(f"→ Wrote pair usage counts to {count_csv!r}")


if __name__ == '__main__':
    region_dir = os.path.join(input_root, region)
    out_dir    = os.path.join(output_root, region)
    os.makedirs(out_dir, exist_ok=True)

    stats_csv  = os.path.join(out_dir, f"batch_{region}_pcf_stats.csv")
    counts_csv = os.path.join(out_dir, f"batch_{region}_pair_counts.csv")

    print(f"\nProcessing region: {region}")
    average_and_sd_over_all_rois(region_dir, stats_csv, counts_csv, fill_missing=0.0)


# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── USER SETTINGS ──────────────────────────────────────────
region         = "Healthy"
stats_csv      = "/Volumes/My Passport/Spatial_Proteomics_data_final/Averaged_ROIs_Per_Region/PCF/Healthy/batch_Healthy_pcf_stats.csv"
counts_csv     = "/Volumes/My Passport/Spatial_Proteomics_data_final/Averaged_ROIs_Per_Region/PCF/Healthy/batch_Healthy_pair_counts.csv"
save_plots_to  = os.path.join(os.path.dirname(stats_csv), "Individual_Pair_Plots")
os.makedirs(save_plots_to, exist_ok=True)

# color map for plotting
color_map = {"Healthy": "lightgreen"}

# load stats and counts
df_stats = pd.read_csv(stats_csv)
df_counts = pd.read_csv(counts_csv).set_index('pair')

# identify all pairs
pair_cols = [col[:-5] for col in df_stats.columns if col.endswith('_mean')]

for pair in sorted(pair_cols):
    # extract R and metrics
    r = df_stats['R'].values
    m = df_stats[f"{pair}_mean"].values
    s = df_stats[f"{pair}_std"].values
    n = df_counts.loc[pair, 'num_rois'] if pair in df_counts.index else np.nan
    # compute 95% CI
    ci = 1.96 * s / np.sqrt(n)

    # plot
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.plot(r, m, label=region, lw=1.5, color=color_map.get(region, 'black'))
    ax.fill_between(r, m-ci, m+ci, alpha=0.2, color=color_map.get(region, 'black'))

    # self-pair highlight
    if '⟷' in pair and pair.split('⟷')[0].strip() == pair.split('⟷')[1].strip():
        ax.set_facecolor('#fff6f6')

    ax.axhline(1, linestyle='--', lw=0.8, color='gray')
    ax.set_ylim(0, None)
    ax.set_xlabel('R (μm)')
    ax.set_ylabel(f"g(r): {pair}", labelpad=5, fontsize=9)
    ax.set_title(pair)
    ax.legend(title='Region')
    fig.tight_layout()

    # save
    out_fn = os.path.join(save_plots_to, f"{pair.replace(' ', '_')}.png")
    fig.savefig(out_fn, dpi=250)
    plt.close(fig)

print(f"✔️ Individual pair plots written to {save_plots_to}")


# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────
regions     = ["Tumour", "Peritumour", "Healthy"]
output_root = "/Volumes/My Passport/Spatial_Proteomics_data_final/Averaged_ROIs_Per_Region/PCF"

# define which pairs of regions to overlay
region_groups = [
    ("Tumour", "Peritumour"),
    ("Peritumour", "Healthy"),
    ("Tumour", "Healthy")
]

# colors for each region
color_map = {
    "Tumour":     "orange",
    "Peritumour": "lightblue",
    "Healthy":    "lightgreen"
}
# ───────────────────────────────────────────────────────────────

# load all stats & counts
data = {}
for reg in regions:
    stats_fn = os.path.join(output_root, reg, f"batch_{reg}_pcf_stats.csv")
    counts_fn = os.path.join(output_root, reg, f"batch_{reg}_pair_counts.csv")
    stats = pd.read_csv(stats_fn)
    counts = pd.read_csv(counts_fn).set_index("pair")["num_rois"].to_dict()
    data[reg] = {"stats": stats, "counts": counts}

# all g(r) pairs present in any region
all_pairs = {
    col[:-5]
    for reg in regions
    for col in data[reg]["stats"].columns
    if col.endswith("_mean")
}

# run through each region pair
for reg1, reg2 in region_groups:
    save_dir = os.path.join(output_root, f"Combined_{reg1}_{reg2}")
    os.makedirs(save_dir, exist_ok=True)

    for pair in sorted(all_pairs):
        # skip if either region lacks this pair
        df1 = data[reg1]["stats"]; df2 = data[reg2]["stats"]
        if f"{pair}_mean" not in df1 or f"{pair}_mean" not in df2:
            continue

        # extract R, means, stds, Ns, and compute 95% CI
        r = df1["R"].values
        m1, s1 = df1[f"{pair}_mean"].values, df1[f"{pair}_std"].values
        n1 = data[reg1]["counts"].get(pair, np.nan)
        ci1 = 1.96 * s1 / np.sqrt(n1)

        m2, s2 = df2[f"{pair}_mean"].values, df2[f"{pair}_std"].values
        n2 = data[reg2]["counts"].get(pair, np.nan)
        ci2 = 1.96 * s2 / np.sqrt(n2)

        # plotting
        fig, ax = plt.subplots(figsize=(7.5,3.5))
        ax.plot(r, m1, lw=1.5, label=reg1, color=color_map[reg1])
        ax.fill_between(r, m1-ci1, m1+ci1, alpha=0.2, color=color_map[reg1])
        ax.plot(r, m2, lw=1.5, label=reg2, color=color_map[reg2])
        ax.fill_between(r, m2-ci2, m2+ci2, alpha=0.2, color=color_map[reg2])

        # highlight self–self
        if "⟷" in pair:
            a, b = [x.strip() for x in pair.split("⟷",1)]
            if a == b:
                ax.set_facecolor("#fff6f6")

        ax.axhline(1, linestyle="--", color="gray", lw=0.8)
        ax.set_ylim(0, None)
        ax.set_xlabel("R (μm)")
        ax.set_ylabel(f"g(r): {pair}", fontsize=6, labelpad=5)
        ax.set_title(pair)
        ax.legend(title="Region")
        fig.tight_layout()

        out_fn = os.path.join(save_dir, f"{pair.replace(' ', '_')}.png")
        fig.savefig(out_fn, dpi=350)
        plt.close(fig)

    print(f"✔️ Plots for Combined_{reg1}_{reg2} saved → {save_dir}")


# In[ ]:


# 2.3.2 Averaged cross-PCF with Region-Specific ROI Exclusion


# In[ ]:


import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
#   USER SETTINGS
# =========================

regions = ["Tumour", "Peritumour", "Healthy"]
exclude_type = "Immune Cells"   # CHANGE THIS for other exclusion types

input_root = "/Volumes/My Passport/Spatial_Proteomics_data_final/PCF"
exclude_dir = "/Volumes/My Passport/Spatial_Proteomics_data_final"

# OUTPUT: new results folder for this exclusion
main_out_root = "/Volumes/My Passport/Spatial_Proteomics_data_final/Averaged_ROIs_Per_Region"
filter_folder = f"PCF_Filter_{exclude_type.replace(' ', '_')}"
output_root   = os.path.join(main_out_root, filter_folder)
os.makedirs(output_root, exist_ok=True)

color_map = {"Tumour": "orange", "Peritumour": "lightblue", "Healthy": "lightgreen"}

region_pairs = [
    ("Tumour", "Peritumour"),
    ("Peritumour", "Healthy"),
    ("Tumour", "Healthy"),
]

# =========================
#    UTILITY FUNCTIONS
# =========================

def load_excluded_rois(exclude_csv, exclude_type):
    if not os.path.exists(exclude_csv):
        return set()
    df = pd.read_csv(exclude_csv)
    if "broad_type" in df.columns:
        df = df[df["broad_type"] == exclude_type]
    # Defensive: strip whitespace, ensure string type
    return set(
        (str(slide).strip(), str(roi).strip())
        for slide, roi in zip(df['Slide'], df['ROI'])
    )

def get_roi_id_from_path(path):
    bn = os.path.basename(path)
    slide = os.path.basename(os.path.dirname(path))
    roi = bn.replace('.csv', '').replace('_pcf', '')
    return (str(slide).strip(), str(roi).strip())

def filtered_roi_files(region_dir, exclude_set):
    all_files = glob.glob(os.path.join(region_dir, "Slide_*", "ROI_*_pcf.csv"))
    return [fn for fn in all_files if get_roi_id_from_path(fn) not in exclude_set], all_files

# =========================
#   PATCHED AGGREGATION
# =========================

def average_and_sd_over_all_rois(region_dir, out_csv, count_csv, fill_missing=0.0, roi_files=None):
    from collections import Counter

    if roi_files is None:
        roi_files = glob.glob(os.path.join(region_dir, "Slide_*", "ROI_*_pcf.csv"))
    if not roi_files:
        raise RuntimeError(f"No ROI PCF CSVs found in {region_dir!r}")

    rois = []
    pair_counts = Counter()
    for fn in sorted(roi_files):
        df = pd.read_csv(fn)
        rois.append(df)
        for col in df.columns.drop('R'):
            pair_counts[col] += 1

    valid_pairs = sorted(pair_counts.keys())
    r_vals = rois[0]['R'].values
    out_df = pd.DataFrame({'R': r_vals})

    n_rois = len(rois)

    for pair in valid_pairs:
        arrs = []
        for df in rois:
            if pair in df.columns:
                arrs.append(df[pair].values)
            else:
                arrs.append(np.full_like(r_vals, fill_missing, dtype=float))
        mat = np.vstack(arrs)

        if np.isnan(fill_missing):
            mean = np.nanmean(mat, axis=0)
            std  = np.nanstd(mat, axis=0, ddof=0)
        else:
            mean = mat.mean(axis=0)
            std  = mat.std(axis=0, ddof=0)

        out_df[f"{pair}_mean"] = mean
        out_df[f"{pair}_std"]  = std

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"→ Wrote PCF mean/std matrix to {out_csv!r}")

    summary = pd.DataFrame({
        'pair': valid_pairs,
        'num_rois': [pair_counts[p] for p in valid_pairs],
        'total_rois': n_rois
    })
    summary.to_csv(count_csv, index=False)
    print(f"→ Wrote pair usage counts to {count_csv!r}")

# =========================
#     MAIN PIPELINE
# =========================

for region in regions:
    print(f"\n--- Processing region: {region} (excluding {exclude_type}) ---")
    region_dir = os.path.join(input_root, region)
    exclude_csv = os.path.join(exclude_dir, f"Exclude_{region.replace(' ','_')}_rois.csv")
    exclude_set = load_excluded_rois(exclude_csv, exclude_type)
    filtered_files, all_files = filtered_roi_files(region_dir, exclude_set)
    n_excluded = len([f for f in all_files if get_roi_id_from_path(f) in exclude_set])
    print(f"  Excluded {n_excluded} of {len(all_files)} ROIs for {region}.")
    if n_excluded > 0:
        print("   Excluded ROIs:")
        for slide, roi in sorted(exclude_set):
            print(f"    {slide}, {roi}")

    # Output file names, in the new folder
    out_dir = os.path.join(output_root, region)
    os.makedirs(out_dir, exist_ok=True)
    stats_csv  = os.path.join(out_dir, f"batch_{region}_pcf_stats.csv")
    counts_csv = os.path.join(out_dir, f"batch_{region}_pair_counts.csv")

    # Averaging with filtered ROIs
    average_and_sd_over_all_rois(region_dir, stats_csv, counts_csv, fill_missing=0.0, roi_files=filtered_files)

    # ---- INDIVIDUAL REGION PLOTS ----
    save_plots_to = os.path.join(out_dir, "Individual_Pair_Plots")
    os.makedirs(save_plots_to, exist_ok=True)
    df_stats = pd.read_csv(stats_csv)
    df_counts = pd.read_csv(counts_csv).set_index('pair')
    pair_cols = [col[:-5] for col in df_stats.columns if col.endswith('_mean')]

    for pair in sorted(pair_cols):
        r = df_stats['R'].values
        m = df_stats[f"{pair}_mean"].values
        s = df_stats[f"{pair}_std"].values
        n = df_counts.loc[pair, 'num_rois'] if pair in df_counts.index else np.nan
        ci = 1.96 * s / np.sqrt(n) if n > 0 else np.zeros_like(s)
        fig, ax = plt.subplots(figsize=(7.5, 3.5))
        ax.plot(r, m, label=region, lw=1.5, color=color_map.get(region, 'black'))
        ax.fill_between(r, m-ci, m+ci, alpha=0.2, color=color_map.get(region, 'black'))
        if '⟷' in pair and pair.split('⟷')[0].strip() == pair.split('⟷')[1].strip():
            ax.set_facecolor('#fff6f6')
        ax.axhline(1, linestyle='--', lw=0.8, color='gray')
        ax.set_ylim(0, None)
        ax.set_xlabel('R (μm)')
        ax.set_ylabel(f"g(r): {pair}", labelpad=5, fontsize=9)
        ax.set_title(pair)
        ax.legend(title='Region')
        fig.tight_layout()
        out_fn = os.path.join(save_plots_to, f"{pair.replace(' ', '_')}.png")
        fig.savefig(out_fn, dpi=250)
        plt.close(fig)
    print(f"  ✔️ Individual pair plots written to {save_plots_to}")
    
# ---- Save exclusion summary for all regions ----
summary_records = []
for region in regions:
    region_dir = os.path.join(input_root, region)
    exclude_csv = os.path.join(exclude_dir, f"Exclude_{region.replace(' ','_')}_rois.csv")
    exclude_set = load_excluded_rois(exclude_csv, exclude_type)
    filtered_files, all_files = filtered_roi_files(region_dir, exclude_set)
    excluded_tuples = [get_roi_id_from_path(f) for f in all_files if get_roi_id_from_path(f) in exclude_set]
    n_excluded = len(excluded_tuples)
    summary_records.append({
        "region": region,
        "n_excluded": n_excluded,
        "total_rois": len(all_files),
        "excluded_pairs": "; ".join([f"{s}|{r}" for s, r in excluded_tuples]) if n_excluded > 0 else "",
    })

summary_df = pd.DataFrame(summary_records)
summary_csv = os.path.join(output_root, f"Exclusion_Summary_{exclude_type.replace(' ', '_')}.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"\nExclusion summary saved: {summary_csv}")

# ---- OVERLAY (PAIRWISE REGION) PLOTS ----

# Load all region stats/counts (for this exclusion version)
data = {}
for region in regions:
    out_dir = os.path.join(output_root, region)
    stats_csv  = os.path.join(out_dir, f"batch_{region}_pcf_stats.csv")
    counts_csv = os.path.join(out_dir, f"batch_{region}_pair_counts.csv")
    stats = pd.read_csv(stats_csv)
    counts = pd.read_csv(counts_csv).set_index("pair")["num_rois"].to_dict()
    data[region] = {"stats": stats, "counts": counts}

all_pairs = {
    col[:-5]
    for region in regions
    for col in data[region]["stats"].columns
    if col.endswith("_mean")
}

for reg1, reg2 in region_pairs:
    save_dir = os.path.join(output_root, f"Combined_{reg1}_vs_{reg2}")
    os.makedirs(save_dir, exist_ok=True)
    for pair in sorted(all_pairs):
        df1 = data[reg1]["stats"]; df2 = data[reg2]["stats"]
        if f"{pair}_mean" not in df1 or f"{pair}_mean" not in df2:
            continue
        r = df1["R"].values
        m1, s1 = df1[f"{pair}_mean"].values, df1[f"{pair}_std"].values
        n1 = data[reg1]["counts"].get(pair, np.nan)
        ci1 = 1.96 * s1 / np.sqrt(n1) if n1 > 0 else np.zeros_like(s1)
        m2, s2 = df2[f"{pair}_mean"].values, df2[f"{pair}_std"].values
        n2 = data[reg2]["counts"].get(pair, np.nan)
        ci2 = 1.96 * s2 / np.sqrt(n2) if n2 > 0 else np.zeros_like(s2)
        fig, ax = plt.subplots(figsize=(7.5,3.5))
        ax.plot(r, m1, lw=1.5, label=reg1, color=color_map[reg1])
        ax.fill_between(r, m1-ci1, m1+ci1, alpha=0.2, color=color_map[reg1])
        ax.plot(r, m2, lw=1.5, label=reg2, color=color_map[reg2])
        ax.fill_between(r, m2-ci2, m2+ci2, alpha=0.2, color=color_map[reg2])
        if "⟷" in pair:
            a, b = [x.strip() for x in pair.split("⟷",1)]
            if a == b:
                ax.set_facecolor("#fff6f6")
        ax.axhline(1, linestyle="--", color="gray", lw=0.8)
        ax.set_ylim(0, None)
        ax.set_xlabel("R (μm)")
        ax.set_ylabel(f"g(r): {pair}", fontsize=6, labelpad=5)
        ax.set_title(pair)
        ax.legend(title="Region")
        fig.tight_layout()
        out_fn = os.path.join(save_dir, f"{pair.replace(' ', '_')}.png")
        fig.savefig(out_fn, dpi=350)
        plt.close(fig)
    print(f"  ✔️ Overlay plots for {reg1} vs {reg2} saved → {save_dir}")

print(f"\n✅ All results for exclusion version: {exclude_type} are in\n{output_root}\n")


# In[ ]:


import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
#   USER SETTINGS
# =========================

regions = ["Tumour", "Peritumour", "Healthy"]
exclude_type = "CAFs"   # CHANGE THIS for other exclusion types

input_root = "/Volumes/My Passport/Spatial_Proteomics_data_final/PCF"
exclude_dir = "/Volumes/My Passport/Spatial_Proteomics_data_final"

# OUTPUT: new results folder for this exclusion
main_out_root = "/Volumes/My Passport/Spatial_Proteomics_data_final/Averaged_ROIs_Per_Region"
filter_folder = f"PCF_Filter_{exclude_type.replace(' ', '_')}"
output_root   = os.path.join(main_out_root, filter_folder)
os.makedirs(output_root, exist_ok=True)

color_map = {"Tumour": "orange", "Peritumour": "lightblue", "Healthy": "lightgreen"}

region_pairs = [
    ("Tumour", "Peritumour"),
    ("Peritumour", "Healthy"),
    ("Tumour", "Healthy"),
]

# =========================
#    UTILITY FUNCTIONS
# =========================

def load_excluded_rois(exclude_csv, exclude_type):
    if not os.path.exists(exclude_csv):
        return set()
    df = pd.read_csv(exclude_csv)
    if "broad_type" in df.columns:
        df = df[df["broad_type"] == exclude_type]
    # Defensive: strip whitespace, ensure string type
    return set(
        (str(slide).strip(), str(roi).strip())
        for slide, roi in zip(df['Slide'], df['ROI'])
    )

def get_roi_id_from_path(path):
    bn = os.path.basename(path)
    slide = os.path.basename(os.path.dirname(path))
    roi = bn.replace('.csv', '').replace('_pcf', '')
    return (str(slide).strip(), str(roi).strip())

def filtered_roi_files(region_dir, exclude_set):
    all_files = glob.glob(os.path.join(region_dir, "Slide_*", "ROI_*_pcf.csv"))
    return [fn for fn in all_files if get_roi_id_from_path(fn) not in exclude_set], all_files

# =========================
#   PATCHED AGGREGATION
# =========================

def average_and_sd_over_all_rois(region_dir, out_csv, count_csv, fill_missing=0.0, roi_files=None):
    from collections import Counter

    if roi_files is None:
        roi_files = glob.glob(os.path.join(region_dir, "Slide_*", "ROI_*_pcf.csv"))
    if not roi_files:
        raise RuntimeError(f"No ROI PCF CSVs found in {region_dir!r}")

    rois = []
    pair_counts = Counter()
    for fn in sorted(roi_files):
        df = pd.read_csv(fn)
        rois.append(df)
        for col in df.columns.drop('R'):
            pair_counts[col] += 1

    valid_pairs = sorted(pair_counts.keys())
    r_vals = rois[0]['R'].values
    out_df = pd.DataFrame({'R': r_vals})

    n_rois = len(rois)

    for pair in valid_pairs:
        arrs = []
        for df in rois:
            if pair in df.columns:
                arrs.append(df[pair].values)
            else:
                arrs.append(np.full_like(r_vals, fill_missing, dtype=float))
        mat = np.vstack(arrs)

        if np.isnan(fill_missing):
            mean = np.nanmean(mat, axis=0)
            std  = np.nanstd(mat, axis=0, ddof=0)
        else:
            mean = mat.mean(axis=0)
            std  = mat.std(axis=0, ddof=0)

        out_df[f"{pair}_mean"] = mean
        out_df[f"{pair}_std"]  = std

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"→ Wrote PCF mean/std matrix to {out_csv!r}")

    summary = pd.DataFrame({
        'pair': valid_pairs,
        'num_rois': [pair_counts[p] for p in valid_pairs],
        'total_rois': n_rois
    })
    summary.to_csv(count_csv, index=False)
    print(f"→ Wrote pair usage counts to {count_csv!r}")

# =========================
#     MAIN PIPELINE
# =========================

for region in regions:
    print(f"\n--- Processing region: {region} (excluding {exclude_type}) ---")
    region_dir = os.path.join(input_root, region)
    exclude_csv = os.path.join(exclude_dir, f"Exclude_{region.replace(' ','_')}_rois.csv")
    exclude_set = load_excluded_rois(exclude_csv, exclude_type)
    filtered_files, all_files = filtered_roi_files(region_dir, exclude_set)
    n_excluded = len([f for f in all_files if get_roi_id_from_path(f) in exclude_set])
    print(f"  Excluded {n_excluded} of {len(all_files)} ROIs for {region}.")
    if n_excluded > 0:
        print("   Excluded ROIs:")
        for slide, roi in sorted(exclude_set):
            print(f"    {slide}, {roi}")

    # Output file names, in the new folder
    out_dir = os.path.join(output_root, region)
    os.makedirs(out_dir, exist_ok=True)
    stats_csv  = os.path.join(out_dir, f"batch_{region}_pcf_stats.csv")
    counts_csv = os.path.join(out_dir, f"batch_{region}_pair_counts.csv")

    # Averaging with filtered ROIs
    average_and_sd_over_all_rois(region_dir, stats_csv, counts_csv, fill_missing=0.0, roi_files=filtered_files)

    # ---- INDIVIDUAL REGION PLOTS ----
    save_plots_to = os.path.join(out_dir, "Individual_Pair_Plots")
    os.makedirs(save_plots_to, exist_ok=True)
    df_stats = pd.read_csv(stats_csv)
    df_counts = pd.read_csv(counts_csv).set_index('pair')
    pair_cols = [col[:-5] for col in df_stats.columns if col.endswith('_mean')]

    for pair in sorted(pair_cols):
        r = df_stats['R'].values
        m = df_stats[f"{pair}_mean"].values
        s = df_stats[f"{pair}_std"].values
        n = df_counts.loc[pair, 'num_rois'] if pair in df_counts.index else np.nan
        ci = 1.96 * s / np.sqrt(n) if n > 0 else np.zeros_like(s)
        fig, ax = plt.subplots(figsize=(7.5, 3.5))
        ax.plot(r, m, label=region, lw=1.5, color=color_map.get(region, 'black'))
        ax.fill_between(r, m-ci, m+ci, alpha=0.2, color=color_map.get(region, 'black'))
        if '⟷' in pair and pair.split('⟷')[0].strip() == pair.split('⟷')[1].strip():
            ax.set_facecolor('#fff6f6')
        ax.axhline(1, linestyle='--', lw=0.8, color='gray')
        ax.set_ylim(0, None)
        ax.set_xlabel('R (μm)')
        ax.set_ylabel(f"g(r): {pair}", labelpad=5, fontsize=9)
        ax.set_title(pair)
        ax.legend(title='Region')
        fig.tight_layout()
        out_fn = os.path.join(save_plots_to, f"{pair.replace(' ', '_')}.png")
        fig.savefig(out_fn, dpi=250)
        plt.close(fig)
    print(f"  ✔️ Individual pair plots written to {save_plots_to}")
    
# ---- Save exclusion summary for all regions ----
summary_records = []
for region in regions:
    region_dir = os.path.join(input_root, region)
    exclude_csv = os.path.join(exclude_dir, f"Exclude_{region.replace(' ','_')}_rois.csv")
    exclude_set = load_excluded_rois(exclude_csv, exclude_type)
    filtered_files, all_files = filtered_roi_files(region_dir, exclude_set)
    excluded_tuples = [get_roi_id_from_path(f) for f in all_files if get_roi_id_from_path(f) in exclude_set]
    n_excluded = len(excluded_tuples)
    summary_records.append({
        "region": region,
        "n_excluded": n_excluded,
        "total_rois": len(all_files),
        "excluded_pairs": "; ".join([f"{s}|{r}" for s, r in excluded_tuples]) if n_excluded > 0 else "",
    })

summary_df = pd.DataFrame(summary_records)
summary_csv = os.path.join(output_root, f"Exclusion_Summary_{exclude_type.replace(' ', '_')}.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"\nExclusion summary saved: {summary_csv}")

# ---- OVERLAY (PAIRWISE REGION) PLOTS ----

# Load all region stats/counts (for this exclusion version)
data = {}
for region in regions:
    out_dir = os.path.join(output_root, region)
    stats_csv  = os.path.join(out_dir, f"batch_{region}_pcf_stats.csv")
    counts_csv = os.path.join(out_dir, f"batch_{region}_pair_counts.csv")
    stats = pd.read_csv(stats_csv)
    counts = pd.read_csv(counts_csv).set_index("pair")["num_rois"].to_dict()
    data[region] = {"stats": stats, "counts": counts}

all_pairs = {
    col[:-5]
    for region in regions
    for col in data[region]["stats"].columns
    if col.endswith("_mean")
}

for reg1, reg2 in region_pairs:
    save_dir = os.path.join(output_root, f"Combined_{reg1}_vs_{reg2}")
    os.makedirs(save_dir, exist_ok=True)
    for pair in sorted(all_pairs):
        df1 = data[reg1]["stats"]; df2 = data[reg2]["stats"]
        if f"{pair}_mean" not in df1 or f"{pair}_mean" not in df2:
            continue
        r = df1["R"].values
        m1, s1 = df1[f"{pair}_mean"].values, df1[f"{pair}_std"].values
        n1 = data[reg1]["counts"].get(pair, np.nan)
        ci1 = 1.96 * s1 / np.sqrt(n1) if n1 > 0 else np.zeros_like(s1)
        m2, s2 = df2[f"{pair}_mean"].values, df2[f"{pair}_std"].values
        n2 = data[reg2]["counts"].get(pair, np.nan)
        ci2 = 1.96 * s2 / np.sqrt(n2) if n2 > 0 else np.zeros_like(s2)
        fig, ax = plt.subplots(figsize=(7.5,3.5))
        ax.plot(r, m1, lw=1.5, label=reg1, color=color_map[reg1])
        ax.fill_between(r, m1-ci1, m1+ci1, alpha=0.2, color=color_map[reg1])
        ax.plot(r, m2, lw=1.5, label=reg2, color=color_map[reg2])
        ax.fill_between(r, m2-ci2, m2+ci2, alpha=0.2, color=color_map[reg2])
        if "⟷" in pair:
            a, b = [x.strip() for x in pair.split("⟷",1)]
            if a == b:
                ax.set_facecolor("#fff6f6")
        ax.axhline(1, linestyle="--", color="gray", lw=0.8)
        ax.set_ylim(0, None)
        ax.set_xlabel("R (μm)")
        ax.set_ylabel(f"g(r): {pair}", fontsize=6, labelpad=5)
        ax.set_title(pair)
        ax.legend(title="Region")
        fig.tight_layout()
        out_fn = os.path.join(save_dir, f"{pair.replace(' ', '_')}.png")
        fig.savefig(out_fn, dpi=350)
        plt.close(fig)
    print(f"  ✔️ Overlay plots for {reg1} vs {reg2} saved → {save_dir}")

print(f"\n✅ All results for exclusion version: {exclude_type} are in\n{output_root}\n")


# In[ ]:





# In[ ]:




