#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('writefile', 'step1_new_Trans_TDA_neighbourhood_cluster.py', 'import os\nimport glob\nimport math\nimport pickle\n\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nimport muspan as ms\nfrom muspan.query import return_object_IDs_from_query_like\n\nfrom itertools import combinations\nfrom multiprocessing import Pool\n\n# --------- EDIT THESE FOR EACH RUN -------------\nROOT_IN  = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/Region_with_cluster_csvs/Healthy"\nROOT_OUT = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/TDA_NC/Healthy"\nROI_GLOB = "ROI_*.csv"\nN_PROC   = 5\n# Maximum distance for Vietoris–Rips\nMD = 707\n\n# -----------------------------------------------\n\ndef process_roi(args):\n    csv_path, out_dir = args\n    roi = os.path.splitext(os.path.basename(csv_path))[0]\n    df  = pd.read_csv(csv_path)\n    \n    # relabel clusters as strings like "Cluster_3"\n    df[\'cluster\'] = \'Cluster_\' + df[\'cluster\'].astype(int).astype(str)\n\n    # get sorted cluster IDs as ints\n    cluster_nums = sorted(\n        df[\'cluster\']\n          .str.replace("Cluster_", "")\n          .astype(int)\n          .unique()\n    )\n    # build a list of (axis,category) for each cluster\n    present = [(\'cluster\', f"Cluster_{i}") for i in cluster_nums]\n\n    # convert mm → µm\n    coords = df[[\'x_mm\', \'y_mm\']].to_numpy() * 1000.0\n\n    # build the muspan domain and label it\n    dom = ms.domain(roi)\n    dom.add_points(coords, \'Cell centres\')\n    dom.add_labels(\'cluster\', df[\'cluster\'])\n\n    valid = []\n    # 1) Single‐cluster TDA jobs, only do TDA if at least 3 cells of that cluster are present.\n    for ax, cat in present:\n        ids = return_object_IDs_from_query_like(dom, (ax, cat))\n        if len(ids) >= 3:\n            valid.append(((ax, cat), None, ids))\n\n    # 2) Pairwise‐cluster TDA jobs\n    # Now TDA and diagrams are only computed for a cluster pair \n    # if both clusters are present (at least one cell each) in that ROI.\n    for (ax1, cat1), (ax2, cat2) in combinations(present, 2):\n        i1 = return_object_IDs_from_query_like(dom, (ax1, cat1))\n        i2 = return_object_IDs_from_query_like(dom, (ax2, cat2))\n        if (len(i1) > 0) and (len(i2) > 0):\n            pop = np.concatenate([i1, i2])\n            if len(pop) >= 3:\n                valid.append(((ax1, cat1), (ax2, cat2), pop))\n            \n    if not valid:\n        print(f"[{roi}] no valid clusters or pairs, skipping")\n        return\n\n    # Prepare plotting grid\n    tot  = len(valid)\n    cols = min(6, tot)\n    rows = math.ceil(tot / cols)\n    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 8 * rows))\n    axes = np.atleast_1d(axes).flatten()\n\n    pd_dict = {}\n    for idx, (first, second, pop) in enumerate(valid):\n        pers = ms.topology.vietoris_rips_filtration(\n            dom,\n            population=pop,\n            max_dimension=1,\n            max_distance=MD,\n            visualise_output=False\n        )\n        dgms = pers[\'dgms\']\n\n        # Name key: either a single cluster or a pair\n        if second is None:\n            key = first[1]                 # e.g. "Cluster_3"\n        else:\n            key = f"{first[1]}⟷{second[1]}"  # e.g. "Cluster_2⟷Cluster_5"\n\n        pd_dict[key] = dgms\n\n        # Plot persistence diagram\n        ax = axes[idx]\n        # H₀: blue dots with thin black boundary\n        if dgms[0].size:\n            ax.scatter(\n                dgms[0][:, 0], dgms[0][:, 1],\n                s=70,\n                marker=\'o\',\n                facecolors=\'skyblue\',\n                edgecolors=\'black\',\n                linewidths=0.7,\n                label=\'H₀\'\n            )\n        # H₁: orange dots with thin black boundary\n        if dgms[1].size:\n            ax.scatter(\n                dgms[1][:, 0], dgms[1][:, 1],\n                s=70,\n                marker=\'o\',\n                facecolors=\'orange\',\n                edgecolors=\'black\',\n                linewidths=0.7,\n                label=\'H₁\'\n            )\n        ax.plot([0, MD], [0, MD], linestyle=\':\', color=\'k\')\n        ax.set_title(key, fontsize=16)\n        ax.set_xlabel(\'birth (µm)\', fontsize=14)\n        ax.set_ylabel(\'death (µm)\', fontsize=14)\n        ax.tick_params(labelsize=14)\n        ax.legend(fontsize=12, markerscale=1.85, loc=\'upper left\')\n\n    # Turn off any extra subplots\n    for ax in axes[tot:]:\n        ax.axis(\'off\')\n\n    # Save outputs\n    os.makedirs(out_dir, exist_ok=True)\n    fig.tight_layout()\n    fig.savefig(os.path.join(out_dir, f"{roi}_PDs.png"), dpi=250)\n    plt.close(fig)\n\n    with open(os.path.join(out_dir, f"{roi}_PDs.pkl"), \'wb\') as f:\n        pickle.dump(pd_dict, f)\n\ndef run_slide(slide):\n    in_dir  = os.path.join(ROOT_IN, slide)\n    out_dir = os.path.join(ROOT_OUT, slide)\n    os.makedirs(out_dir, exist_ok=True)\n\n    files = sorted(\n        glob.glob(os.path.join(in_dir, ROI_GLOB)),\n        key=lambda fn: int(os.path.splitext(fn)[0].split(\'_\')[-1])\n    )\n    print(f"[{slide}] {len(files)} ROIs → computing PDs")\n\n    with Pool(processes=N_PROC) as pool:\n        pool.map(process_roi, [(fn, out_dir) for fn in files])\n\n    print(f"[{slide}] done")\n\ndef run_all_slides():\n    slides = sorted(\n        d for d in os.listdir(ROOT_IN)\n        if os.path.isdir(os.path.join(ROOT_IN, d)) and d.startswith("Slide_")\n    )\n    for slide in slides:\n        run_slide(slide)\n\nif __name__ == "__main__":\n    run_all_slides()\n')


# In[ ]:


# Done: Tumour, Peritumour, Healthy


# In[ ]:


# Step. 2 Vectorisation (original + thresholded "significant" summaries).
import os
import glob
import pickle
import numpy as np
import pandas as pd
import muspan as ms

# -------- EDIT THESE FOR EACH RUN -----------
REGION   = "Tumour"    # "Healthy", "Peritumour", etc
ROOT     = "/Volumes/My Passport/Spatial_Transcriptomics_data_final"
PD_ROOT  = os.path.join(ROOT, "TDA_NC", REGION)
OUT_CSV  = os.path.join(PD_ROOT, f"{REGION}_all_ROIs_persistence_features.csv")
# --------------------------------------------

# Thresholds for "significant" bars
H0_THRESHOLD = 50.0  # µm for H0 lifespans
H1_THRESHOLD = 10.0   # µm for H1 lifespans

def summarize_stats(arr):
    if len(arr) == 0:
        return {k: 0.0 for k in ['mean','std','p10','p25','p50','p75','p90']}
    return {
        'mean': np.mean(arr),
        'std': np.std(arr, ddof=1) if len(arr) > 1 else 0.0,
        'p10': np.percentile(arr, 10),
        'p25': np.percentile(arr, 25),
        'p50': np.percentile(arr, 50),
        'p75': np.percentile(arr, 75),
        'p90': np.percentile(arr, 90),
    }

def compute_entropy(lifespans):
    if len(lifespans) == 0:
        return 0.0
    norm = lifespans / np.sum(lifespans)
    return -np.sum(norm * np.log(norm + 1e-12))

def vectorize_thresholded(dgms, h0_thresh=H0_THRESHOLD, h1_thresh=H1_THRESHOLD):
    """
    Only produces thresholded ("significant") summary features for H0 and H1:
      - Keeps bars with lifespan > threshold (dimension-specific)
      - Returns vector and names: e.g., H0_significant_nBars, H0_significant_birth_mean, ..., H0_significant_entropy, etc.
    """
    features = []
    names = []
    for dim_idx, diag in enumerate(dgms):  # H0 (0), H1 (1)
        dim = f"H{dim_idx}"
        thresh = h0_thresh if dim_idx == 0 else h1_thresh

        if diag is None or diag.size == 0:
            # zeros for everything
            features.append(0); names.append(f"{dim}_significant_nBars")
            for group in ['birth','death','lifespan','midpoint']:
                for stat in ['mean','std','p10','p25','p50','p75','p90']:
                    features.append(0.0); names.append(f"{dim}_significant_{group}_{stat}")
            features.append(0.0); names.append(f"{dim}_significant_entropy")
            continue

        births = diag[:,0]
        deaths = diag[:,1]
        lifespans = deaths - births
        midpoints = (births + deaths) / 2.0

        # thresholded mask
        mask = lifespans >= thresh
        sig_births = births[mask]
        sig_deaths = deaths[mask]
        sig_lifespans = lifespans[mask]
        sig_midpoints = midpoints[mask]

        # number of significant bars
        features.append(len(sig_lifespans)); names.append(f"{dim}_significant_nBars")

        # summary stats for each group
        for arr, group in [
            (sig_births, 'birth'),
            (sig_deaths, 'death'),
            (sig_lifespans, 'lifespan'),
            (sig_midpoints, 'midpoint'),
        ]:
            stats = summarize_stats(arr)
            for stat in ['mean','std','p10','p25','p50','p75','p90']:
                features.append(stats[stat])
                names.append(f"{dim}_significant_{group}_{stat}")

        # entropy over significant lifespans
        ent = compute_entropy(sig_lifespans)
        features.append(ent); names.append(f"{dim}_significant_entropy")

    return np.hstack(features), names

def load_and_vectorise(pkl_path):
    """
    Load a ROI’s persistence‐diagram pickle and
    vectorise each pair via:
      1) original Ali-style statistics (via muspan.vectorise_persistence)
      2) thresholded 'significant' summaries for H0/H1
    Returns (feat_vector, feature_names), or (None,None).
    """
    with open(pkl_path, 'rb') as f:
        pd_dict = pickle.load(f)

    vecs = []
    names = []

    for pair in sorted(pd_dict.keys()):
        dgms = pd_dict[pair]
        # original vectorisation (unchanged)
        try:
            feat_orig, stat_names = ms.topology.vectorise_persistence(
                {'dgms': dgms},
                method='statistics'
            )
        except Exception:
            # fallback if structure is raw list/array
            feat_orig, stat_names = ms.topology.vectorise_persistence(
                dgms,
                method='statistics'
            )
        prefixed_orig = [f"{pair}_{n}" for n in stat_names]
        names.extend(prefixed_orig)
        vecs.append(feat_orig)

        # thresholded "significant" extra summaries
        # adapt to the format: dgms may be dict with 'dgms' key or list
        if isinstance(dgms, dict) and 'dgms' in dgms:
            diagram_list = dgms['dgms']
        else:
            diagram_list = dgms  # assume already [H0, H1]

        feat_sig, sig_names = vectorize_thresholded(diagram_list)
        prefixed_sig = [f"{pair}_{n}" for n in sig_names]
        names.extend(prefixed_sig)
        vecs.append(feat_sig)

    if not vecs:
        return None, None

    return np.hstack(vecs), names

if __name__ == "__main__":
    records = []
    all_features = set()

    # Walk every Slide_* under PD_ROOT
    for slide in sorted(os.listdir(PD_ROOT)):
        pd_dir = os.path.join(PD_ROOT, slide)
        if not os.path.isdir(pd_dir):
            continue

        for pkl in sorted(glob.glob(os.path.join(pd_dir, "ROI_*_PDs.pkl"))):
            roi = os.path.basename(pkl).split("_PDs.pkl")[0]
            vec, names = load_and_vectorise(pkl)
            if vec is None or names is None:
                continue

            rec = {"Region": REGION, "Slide": slide, "ROI": roi}
            for name, val in zip(names, vec):
                rec[name] = val
                all_features.add(name)
            records.append(rec)

    if not records:
        raise RuntimeError(f"No ROI features found in {PD_ROOT}—check your PDs folders.")

    feat_cols = sorted(all_features)
    df = pd.DataFrame(records)
    df = df[["Region", "Slide", "ROI"] + feat_cols]

    # Fill missing values with 0 before saving
    df = df.fillna(0)

    # Save
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved feature matrix with shape {df.shape} → {OUT_CSV}")


# In[ ]:


# Step. 2 Vectorisation (original + thresholded "significant" summaries).
import os
import glob
import pickle
import numpy as np
import pandas as pd
import muspan as ms

# -------- EDIT THESE FOR EACH RUN -----------
REGION   = "Peritumour"    # "Healthy", "Peritumour", etc
ROOT     = "/Volumes/My Passport/Spatial_Transcriptomics_data_final"
PD_ROOT  = os.path.join(ROOT, "TDA_NC", REGION)
OUT_CSV  = os.path.join(PD_ROOT, f"{REGION}_all_ROIs_persistence_features.csv")
# --------------------------------------------

# Thresholds for "significant" bars
H0_THRESHOLD = 50.0  # µm for H0 lifespans
H1_THRESHOLD = 10.0   # µm for H1 lifespans

def summarize_stats(arr):
    if len(arr) == 0:
        return {k: 0.0 for k in ['mean','std','p10','p25','p50','p75','p90']}
    return {
        'mean': np.mean(arr),
        'std': np.std(arr, ddof=1) if len(arr) > 1 else 0.0,
        'p10': np.percentile(arr, 10),
        'p25': np.percentile(arr, 25),
        'p50': np.percentile(arr, 50),
        'p75': np.percentile(arr, 75),
        'p90': np.percentile(arr, 90),
    }

def compute_entropy(lifespans):
    if len(lifespans) == 0:
        return 0.0
    norm = lifespans / np.sum(lifespans)
    return -np.sum(norm * np.log(norm + 1e-12))

def vectorize_thresholded(dgms, h0_thresh=H0_THRESHOLD, h1_thresh=H1_THRESHOLD):
    """
    Only produces thresholded ("significant") summary features for H0 and H1:
      - Keeps bars with lifespan > threshold (dimension-specific)
      - Returns vector and names: e.g., H0_significant_nBars, H0_significant_birth_mean, ..., H0_significant_entropy, etc.
    """
    features = []
    names = []
    for dim_idx, diag in enumerate(dgms):  # H0 (0), H1 (1)
        dim = f"H{dim_idx}"
        thresh = h0_thresh if dim_idx == 0 else h1_thresh

        if diag is None or diag.size == 0:
            # zeros for everything
            features.append(0); names.append(f"{dim}_significant_nBars")
            for group in ['birth','death','lifespan','midpoint']:
                for stat in ['mean','std','p10','p25','p50','p75','p90']:
                    features.append(0.0); names.append(f"{dim}_significant_{group}_{stat}")
            features.append(0.0); names.append(f"{dim}_significant_entropy")
            continue

        births = diag[:,0]
        deaths = diag[:,1]
        lifespans = deaths - births
        midpoints = (births + deaths) / 2.0

        # thresholded mask
        mask = lifespans >= thresh
        sig_births = births[mask]
        sig_deaths = deaths[mask]
        sig_lifespans = lifespans[mask]
        sig_midpoints = midpoints[mask]

        # number of significant bars
        features.append(len(sig_lifespans)); names.append(f"{dim}_significant_nBars")

        # summary stats for each group
        for arr, group in [
            (sig_births, 'birth'),
            (sig_deaths, 'death'),
            (sig_lifespans, 'lifespan'),
            (sig_midpoints, 'midpoint'),
        ]:
            stats = summarize_stats(arr)
            for stat in ['mean','std','p10','p25','p50','p75','p90']:
                features.append(stats[stat])
                names.append(f"{dim}_significant_{group}_{stat}")

        # entropy over significant lifespans
        ent = compute_entropy(sig_lifespans)
        features.append(ent); names.append(f"{dim}_significant_entropy")

    return np.hstack(features), names

def load_and_vectorise(pkl_path):
    """
    Load a ROI’s persistence‐diagram pickle and
    vectorise each pair via:
      1) original Ali-style statistics (via muspan.vectorise_persistence)
      2) thresholded 'significant' summaries for H0/H1
    Returns (feat_vector, feature_names), or (None,None).
    """
    with open(pkl_path, 'rb') as f:
        pd_dict = pickle.load(f)

    vecs = []
    names = []

    for pair in sorted(pd_dict.keys()):
        dgms = pd_dict[pair]
        # original vectorisation (unchanged)
        try:
            feat_orig, stat_names = ms.topology.vectorise_persistence(
                {'dgms': dgms},
                method='statistics'
            )
        except Exception:
            # fallback if structure is raw list/array
            feat_orig, stat_names = ms.topology.vectorise_persistence(
                dgms,
                method='statistics'
            )
        prefixed_orig = [f"{pair}_{n}" for n in stat_names]
        names.extend(prefixed_orig)
        vecs.append(feat_orig)

        # thresholded "significant" extra summaries
        # adapt to the format: dgms may be dict with 'dgms' key or list
        if isinstance(dgms, dict) and 'dgms' in dgms:
            diagram_list = dgms['dgms']
        else:
            diagram_list = dgms  # assume already [H0, H1]

        feat_sig, sig_names = vectorize_thresholded(diagram_list)
        prefixed_sig = [f"{pair}_{n}" for n in sig_names]
        names.extend(prefixed_sig)
        vecs.append(feat_sig)

    if not vecs:
        return None, None

    return np.hstack(vecs), names

if __name__ == "__main__":
    records = []
    all_features = set()

    # Walk every Slide_* under PD_ROOT
    for slide in sorted(os.listdir(PD_ROOT)):
        pd_dir = os.path.join(PD_ROOT, slide)
        if not os.path.isdir(pd_dir):
            continue

        for pkl in sorted(glob.glob(os.path.join(pd_dir, "ROI_*_PDs.pkl"))):
            roi = os.path.basename(pkl).split("_PDs.pkl")[0]
            vec, names = load_and_vectorise(pkl)
            if vec is None or names is None:
                continue

            rec = {"Region": REGION, "Slide": slide, "ROI": roi}
            for name, val in zip(names, vec):
                rec[name] = val
                all_features.add(name)
            records.append(rec)

    if not records:
        raise RuntimeError(f"No ROI features found in {PD_ROOT}—check your PDs folders.")

    feat_cols = sorted(all_features)
    df = pd.DataFrame(records)
    df = df[["Region", "Slide", "ROI"] + feat_cols]

    # Fill missing values with 0 before saving
    df = df.fillna(0)

    # Save
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved feature matrix with shape {df.shape} → {OUT_CSV}")


# In[ ]:


# Step. 2 Vectorisation (original + thresholded "significant" summaries).
import os
import glob
import pickle
import numpy as np
import pandas as pd
import muspan as ms

# -------- EDIT THESE FOR EACH RUN -----------
REGION   = "Healthy"    # "Healthy", "Peritumour", etc
ROOT     = "/Volumes/My Passport/Spatial_Transcriptomics_data_final"
PD_ROOT  = os.path.join(ROOT, "TDA_NC", REGION)
OUT_CSV  = os.path.join(PD_ROOT, f"{REGION}_all_ROIs_persistence_features.csv")
# --------------------------------------------

# Thresholds for "significant" bars
H0_THRESHOLD = 50.0  # µm for H0 lifespans
H1_THRESHOLD = 10.0   # µm for H1 lifespans

def summarize_stats(arr):
    if len(arr) == 0:
        return {k: 0.0 for k in ['mean','std','p10','p25','p50','p75','p90']}
    return {
        'mean': np.mean(arr),
        'std': np.std(arr, ddof=1) if len(arr) > 1 else 0.0,
        'p10': np.percentile(arr, 10),
        'p25': np.percentile(arr, 25),
        'p50': np.percentile(arr, 50),
        'p75': np.percentile(arr, 75),
        'p90': np.percentile(arr, 90),
    }

def compute_entropy(lifespans):
    if len(lifespans) == 0:
        return 0.0
    norm = lifespans / np.sum(lifespans)
    return -np.sum(norm * np.log(norm + 1e-12))

def vectorize_thresholded(dgms, h0_thresh=H0_THRESHOLD, h1_thresh=H1_THRESHOLD):
    """
    Only produces thresholded ("significant") summary features for H0 and H1:
      - Keeps bars with lifespan > threshold (dimension-specific)
      - Returns vector and names: e.g., H0_significant_nBars, H0_significant_birth_mean, ..., H0_significant_entropy, etc.
    """
    features = []
    names = []
    for dim_idx, diag in enumerate(dgms):  # H0 (0), H1 (1)
        dim = f"H{dim_idx}"
        thresh = h0_thresh if dim_idx == 0 else h1_thresh

        if diag is None or diag.size == 0:
            # zeros for everything
            features.append(0); names.append(f"{dim}_significant_nBars")
            for group in ['birth','death','lifespan','midpoint']:
                for stat in ['mean','std','p10','p25','p50','p75','p90']:
                    features.append(0.0); names.append(f"{dim}_significant_{group}_{stat}")
            features.append(0.0); names.append(f"{dim}_significant_entropy")
            continue

        births = diag[:,0]
        deaths = diag[:,1]
        lifespans = deaths - births
        midpoints = (births + deaths) / 2.0

        # thresholded mask
        mask = lifespans >= thresh
        sig_births = births[mask]
        sig_deaths = deaths[mask]
        sig_lifespans = lifespans[mask]
        sig_midpoints = midpoints[mask]

        # number of significant bars
        features.append(len(sig_lifespans)); names.append(f"{dim}_significant_nBars")

        # summary stats for each group
        for arr, group in [
            (sig_births, 'birth'),
            (sig_deaths, 'death'),
            (sig_lifespans, 'lifespan'),
            (sig_midpoints, 'midpoint'),
        ]:
            stats = summarize_stats(arr)
            for stat in ['mean','std','p10','p25','p50','p75','p90']:
                features.append(stats[stat])
                names.append(f"{dim}_significant_{group}_{stat}")

        # entropy over significant lifespans
        ent = compute_entropy(sig_lifespans)
        features.append(ent); names.append(f"{dim}_significant_entropy")

    return np.hstack(features), names

def load_and_vectorise(pkl_path):
    """
    Load a ROI’s persistence‐diagram pickle and
    vectorise each pair via:
      1) original Ali-style statistics (via muspan.vectorise_persistence)
      2) thresholded 'significant' summaries for H0/H1
    Returns (feat_vector, feature_names), or (None,None).
    """
    with open(pkl_path, 'rb') as f:
        pd_dict = pickle.load(f)

    vecs = []
    names = []

    for pair in sorted(pd_dict.keys()):
        dgms = pd_dict[pair]
        # original vectorisation (unchanged)
        try:
            feat_orig, stat_names = ms.topology.vectorise_persistence(
                {'dgms': dgms},
                method='statistics'
            )
        except Exception:
            # fallback if structure is raw list/array
            feat_orig, stat_names = ms.topology.vectorise_persistence(
                dgms,
                method='statistics'
            )
        prefixed_orig = [f"{pair}_{n}" for n in stat_names]
        names.extend(prefixed_orig)
        vecs.append(feat_orig)

        # thresholded "significant" extra summaries
        # adapt to the format: dgms may be dict with 'dgms' key or list
        if isinstance(dgms, dict) and 'dgms' in dgms:
            diagram_list = dgms['dgms']
        else:
            diagram_list = dgms  # assume already [H0, H1]

        feat_sig, sig_names = vectorize_thresholded(diagram_list)
        prefixed_sig = [f"{pair}_{n}" for n in sig_names]
        names.extend(prefixed_sig)
        vecs.append(feat_sig)

    if not vecs:
        return None, None

    return np.hstack(vecs), names

if __name__ == "__main__":
    records = []
    all_features = set()

    # Walk every Slide_* under PD_ROOT
    for slide in sorted(os.listdir(PD_ROOT)):
        pd_dir = os.path.join(PD_ROOT, slide)
        if not os.path.isdir(pd_dir):
            continue

        for pkl in sorted(glob.glob(os.path.join(pd_dir, "ROI_*_PDs.pkl"))):
            roi = os.path.basename(pkl).split("_PDs.pkl")[0]
            vec, names = load_and_vectorise(pkl)
            if vec is None or names is None:
                continue

            rec = {"Region": REGION, "Slide": slide, "ROI": roi}
            for name, val in zip(names, vec):
                rec[name] = val
                all_features.add(name)
            records.append(rec)

    if not records:
        raise RuntimeError(f"No ROI features found in {PD_ROOT}—check your PDs folders.")

    feat_cols = sorted(all_features)
    df = pd.DataFrame(records)
    df = df[["Region", "Slide", "ROI"] + feat_cols]

    # Fill missing values with 0 before saving
    df = df.fillna(0)

    # Save
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved feature matrix with shape {df.shape} → {OUT_CSV}")


# In[ ]:


# Step 3. Data extraction


# In[ ]:


import os
import re
import pandas as pd

# ---- EDIT HERE ----
REGION     = "Tumour"   # or "Healthy", "Peritumour", etc.
ROOT       = "/Volumes/My Passport/Spatial_Transcriptomics_data_final"
FEAT_CSV   = os.path.join(
    ROOT, "TDA_NC", REGION,
    f"{REGION}_all_ROIs_persistence_features.csv"
)
OUT_CSV    = os.path.join(
    ROOT, "TDA_NC", REGION,
    f"{REGION}_all_ROIs_persistence_summary_clusters.csv"
)
EXCLUDED_ROIS_CSV = os.path.join(
    ROOT, "TDA_NC", REGION,
    f"{REGION}_excluded_rois_no_target_H0_nBars.csv"
)
# -------------------

# 1) Load full feature matrix
df = pd.read_csv(FEAT_CSV)

# 2) Always keep these metadata cols
meta = ["Region", "Slide", "ROI"]

# 3) Target clusters / pairs
selected_clusters = {"Cluster_0", "Cluster_2", "Cluster_7"}
# pairs to consider: 
pair_prefixes = [
    "Cluster_0⟷Cluster_2",
    "Cluster_0⟷Cluster_7",
    "Cluster_2⟷Cluster_7"
]
single_prefixes = [f"Cluster_{i}" for i in [0,2,7]]
all_prefixes = single_prefixes + pair_prefixes

# 4) Build list of original H0 nBars columns for those prefixes (tolerant to naming variants)
orig_h0_patterns = []
for prefix in all_prefixes:
    # matches e.g. Cluster_7_H0 nBars, Cluster_7_H0_nBars, Cluster_7_H0_nBars_orig
    pat = re.compile(rf"^{re.escape(prefix)}_H0(?:[_ ]?nBars)(?:_?orig)?$")
    orig_h0_patterns.append(pat)

orig_h0_cols = []
for c in df.columns:
    if any(pat.match(c) for pat in orig_h0_patterns):
        # be sure this is the unthresholded one (exclude "significant")
        if "significant" in c:
            continue
        orig_h0_cols.append(c)
if not orig_h0_cols:
    raise RuntimeError("No original H0 nBars columns found for the target prefixes. Check naming conventions.")

# 5) ROI filtration: keep ROIs where any of the target original H0 nBars is > 0
mask_keep = (df[orig_h0_cols].fillna(0) > 0).any(axis=1)
df_kept = df.loc[mask_keep].reset_index(drop=True)
df_excluded = df.loc[~mask_keep, ["Region", "Slide", "ROI"]].copy()

# Save excluded ROIs
os.makedirs(os.path.dirname(EXCLUDED_ROIS_CSV), exist_ok=True)
df_excluded.to_csv(EXCLUDED_ROIS_CSV, index=False)
print(f"[{REGION}] excluded {len(df_excluded)} ROIs with no target original H0 nBars > 0 → saved list to {EXCLUDED_ROIS_CSV}")
print(f"[{REGION}] keeping {len(df_kept)} ROIs (dropped {len(df_excluded)})")

# 6) Now selection of thresholded (“significant”) stats for allowed clusters/pairs
selected_params = [
    "H0_significant_nBars", "H1_significant_nBars",
    "H0_significant_lifespan_mean", "H1_significant_lifespan_mean",
    "H0_significant_lifespan_std",  "H1_significant_lifespan_std",
    "H1_significant_birth_mean",  "H1_significant_birth_std"
]

def is_selected(col):
    if col in meta:
        return True

    match = next((p for p in selected_params if col.endswith(f"_{p}")), None)
    if not match:
        return False

    prefix = col[: -(len(match) + 1)]
    if "⟷" in prefix:
        left, right = prefix.split("⟷", 1)
        return (left in selected_clusters) and (right in selected_clusters)
    else:
        return prefix in selected_clusters

keep_cols = [c for c in df_kept.columns if is_selected(c)]
df_trim = df_kept[keep_cols].copy()

# ─── (NEW) Add +1 to H0_significant_nBars columns ──────────────
for col in df_trim.columns:
    if col.endswith("H0_significant_nBars"):
        df_trim[col] = df_trim[col] + 1

# 7) Save trimmed feature matrix
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df_trim.to_csv(OUT_CSV, index=False)
print(f"Dropped {df.shape[1] - df_trim.shape[1]} cols, wrote {df_trim.shape} →\n  {OUT_CSV}")


# In[ ]:


import os
import re
import pandas as pd

# ---- EDIT HERE ----
REGION     = "Peritumour"   # or "Healthy", "Peritumour", etc.
ROOT       = "/Volumes/My Passport/Spatial_Transcriptomics_data_final"
FEAT_CSV   = os.path.join(
    ROOT, "TDA_NC", REGION,
    f"{REGION}_all_ROIs_persistence_features.csv"
)
OUT_CSV    = os.path.join(
    ROOT, "TDA_NC", REGION,
    f"{REGION}_all_ROIs_persistence_summary_clusters.csv"
)
EXCLUDED_ROIS_CSV = os.path.join(
    ROOT, "TDA_NC", REGION,
    f"{REGION}_excluded_rois_no_target_H0_nBars.csv"
)
# -------------------

# 1) Load full feature matrix
df = pd.read_csv(FEAT_CSV)

# 2) Always keep these metadata cols
meta = ["Region", "Slide", "ROI"]

# 3) Target clusters / pairs
selected_clusters = {"Cluster_0", "Cluster_2", "Cluster_7"}
# pairs to consider: 
pair_prefixes = [
    "Cluster_0⟷Cluster_2",
    "Cluster_0⟷Cluster_7",
    "Cluster_2⟷Cluster_7"
]
single_prefixes = [f"Cluster_{i}" for i in [0,2,7]]
all_prefixes = single_prefixes + pair_prefixes

# 4) Build list of original H0 nBars columns for those prefixes (tolerant to naming variants)
orig_h0_patterns = []
for prefix in all_prefixes:
    # matches e.g. Cluster_7_H0 nBars, Cluster_7_H0_nBars, Cluster_7_H0_nBars_orig
    pat = re.compile(rf"^{re.escape(prefix)}_H0(?:[_ ]?nBars)(?:_?orig)?$")
    orig_h0_patterns.append(pat)

orig_h0_cols = []
for c in df.columns:
    if any(pat.match(c) for pat in orig_h0_patterns):
        # be sure this is the unthresholded one (exclude "significant")
        if "significant" in c:
            continue
        orig_h0_cols.append(c)
if not orig_h0_cols:
    raise RuntimeError("No original H0 nBars columns found for the target prefixes. Check naming conventions.")

# 5) ROI filtration: keep ROIs where any of the target original H0 nBars is > 0
mask_keep = (df[orig_h0_cols].fillna(0) > 0).any(axis=1)
df_kept = df.loc[mask_keep].reset_index(drop=True)
df_excluded = df.loc[~mask_keep, ["Region", "Slide", "ROI"]].copy()

# Save excluded ROIs
os.makedirs(os.path.dirname(EXCLUDED_ROIS_CSV), exist_ok=True)
df_excluded.to_csv(EXCLUDED_ROIS_CSV, index=False)
print(f"[{REGION}] excluded {len(df_excluded)} ROIs with no target original H0 nBars > 0 → saved list to {EXCLUDED_ROIS_CSV}")
print(f"[{REGION}] keeping {len(df_kept)} ROIs (dropped {len(df_excluded)})")

# 6) Now selection of thresholded (“significant”) stats for allowed clusters/pairs
selected_params = [
    "H0_significant_nBars", "H1_significant_nBars",
    "H0_significant_lifespan_mean", "H1_significant_lifespan_mean",
    "H0_significant_lifespan_std",  "H1_significant_lifespan_std",
    "H1_significant_birth_mean",  "H1_significant_birth_std"
]

def is_selected(col):
    if col in meta:
        return True

    match = next((p for p in selected_params if col.endswith(f"_{p}")), None)
    if not match:
        return False

    prefix = col[: -(len(match) + 1)]
    if "⟷" in prefix:
        left, right = prefix.split("⟷", 1)
        return (left in selected_clusters) and (right in selected_clusters)
    else:
        return prefix in selected_clusters

keep_cols = [c for c in df_kept.columns if is_selected(c)]
df_trim = df_kept[keep_cols].copy()

# ─── (NEW) Add +1 to H0_significant_nBars columns ──────────────
for col in df_trim.columns:
    if col.endswith("H0_significant_nBars"):
        df_trim[col] = df_trim[col] + 1

# 7) Save trimmed feature matrix
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df_trim.to_csv(OUT_CSV, index=False)
print(f"Dropped {df.shape[1] - df_trim.shape[1]} cols, wrote {df_trim.shape} →\n  {OUT_CSV}")


# In[ ]:


import os
import re
import pandas as pd

# ---- EDIT HERE ----
REGION     = "Healthy"   # or "Healthy", "Peritumour", etc.
ROOT       = "/Volumes/My Passport/Spatial_Transcriptomics_data_final"
FEAT_CSV   = os.path.join(
    ROOT, "TDA_NC", REGION,
    f"{REGION}_all_ROIs_persistence_features.csv"
)
OUT_CSV    = os.path.join(
    ROOT, "TDA_NC", REGION,
    f"{REGION}_all_ROIs_persistence_summary_clusters.csv"
)
EXCLUDED_ROIS_CSV = os.path.join(
    ROOT, "TDA_NC", REGION,
    f"{REGION}_excluded_rois_no_target_H0_nBars.csv"
)
# -------------------

# 1) Load full feature matrix
df = pd.read_csv(FEAT_CSV)

# 2) Always keep these metadata cols
meta = ["Region", "Slide", "ROI"]

# 3) Target clusters / pairs
selected_clusters = {"Cluster_0", "Cluster_2", "Cluster_7"}
# pairs to consider: 
pair_prefixes = [
    "Cluster_0⟷Cluster_2",
    "Cluster_0⟷Cluster_7",
    "Cluster_2⟷Cluster_7"
]
single_prefixes = [f"Cluster_{i}" for i in [0,2,7]]
all_prefixes = single_prefixes + pair_prefixes

# 4) Build list of original H0 nBars columns for those prefixes (tolerant to naming variants)
orig_h0_patterns = []
for prefix in all_prefixes:
    # matches e.g. Cluster_7_H0 nBars, Cluster_7_H0_nBars, Cluster_7_H0_nBars_orig
    pat = re.compile(rf"^{re.escape(prefix)}_H0(?:[_ ]?nBars)(?:_?orig)?$")
    orig_h0_patterns.append(pat)

orig_h0_cols = []
for c in df.columns:
    if any(pat.match(c) for pat in orig_h0_patterns):
        # be sure this is the unthresholded one (exclude "significant")
        if "significant" in c:
            continue
        orig_h0_cols.append(c)
if not orig_h0_cols:
    raise RuntimeError("No original H0 nBars columns found for the target prefixes. Check naming conventions.")

# 5) ROI filtration: keep ROIs where any of the target original H0 nBars is > 0
mask_keep = (df[orig_h0_cols].fillna(0) > 0).any(axis=1)
df_kept = df.loc[mask_keep].reset_index(drop=True)
df_excluded = df.loc[~mask_keep, ["Region", "Slide", "ROI"]].copy()

# Save excluded ROIs
os.makedirs(os.path.dirname(EXCLUDED_ROIS_CSV), exist_ok=True)
df_excluded.to_csv(EXCLUDED_ROIS_CSV, index=False)
print(f"[{REGION}] excluded {len(df_excluded)} ROIs with no target original H0 nBars > 0 → saved list to {EXCLUDED_ROIS_CSV}")
print(f"[{REGION}] keeping {len(df_kept)} ROIs (dropped {len(df_excluded)})")

# 6) Now selection of thresholded (“significant”) stats for allowed clusters/pairs
selected_params = [
    "H0_significant_nBars", "H1_significant_nBars",
    "H0_significant_lifespan_mean", "H1_significant_lifespan_mean",
    "H0_significant_lifespan_std",  "H1_significant_lifespan_std",
    "H1_significant_birth_mean",  "H1_significant_birth_std"
]

def is_selected(col):
    if col in meta:
        return True

    match = next((p for p in selected_params if col.endswith(f"_{p}")), None)
    if not match:
        return False

    prefix = col[: -(len(match) + 1)]
    if "⟷" in prefix:
        left, right = prefix.split("⟷", 1)
        return (left in selected_clusters) and (right in selected_clusters)
    else:
        return prefix in selected_clusters

keep_cols = [c for c in df_kept.columns if is_selected(c)]
df_trim = df_kept[keep_cols].copy()

# ─── (NEW) Add +1 to H0_significant_nBars columns ──────────────
for col in df_trim.columns:
    if col.endswith("H0_significant_nBars"):
        df_trim[col] = df_trim[col] + 1

# 7) Save trimmed feature matrix
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df_trim.to_csv(OUT_CSV, index=False)
print(f"Dropped {df.shape[1] - df_trim.shape[1]} cols, wrote {df_trim.shape} →\n  {OUT_CSV}")


# In[ ]:


# Step 4. Correlation Assessment


# In[ ]:


import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

# ---- CONFIG ----
ROOT = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/TDA_NC"
REGIONS = ["Tumour", "Peritumour", "Healthy"]
# These are your single clusters; pairwise will be built from these
SINGLES = ["Cluster_0", "Cluster_2", "Cluster_7"]
# Output path (aggregate Spearman results)
OUT_CSV = os.path.join(ROOT, "all_regions_slide_027stats_spearman_FDR.csv")
# ----------------

# 1) Load & concatenate the trimmed summary files (assumed to contain only thresholded/significant stats)
roi_dfs = []
for region in REGIONS:
    fn = os.path.join(ROOT, region, f"{region}_all_ROIs_persistence_summary_clusters.csv")
    if not os.path.exists(fn):
        raise FileNotFoundError(f"Expected trimmed summary file not found: {fn}")
    df = pd.read_csv(fn)
    df["Region"] = region  # ensure region column exists / is consistent
    roi_dfs.append(df)
roi_df = pd.concat(roi_dfs, ignore_index=True)

# 2) Slide-level aggregation (mean over ROIs in each Slide within Region)
#    Keep Region and Slide as grouping keys; drop ROI because we aggregate
agg = (
    roi_df
    .drop(columns=["ROI"], errors="ignore")
    .groupby(["Region", "Slide"], observed=True)
    .mean(numeric_only=True)
    .reset_index()
)

# 3) Build allowed prefixes: singles + unordered pairwise
pairwise = [f"{a}⟷{b}" for a, b in itertools.combinations(SINGLES, 2)]
prefixes = SINGLES + pairwise

# 4) Select the thresholded feature columns corresponding to your 80 features:
#    Expect names like "Cluster_0_H0_significant_nBars", "Cluster_4⟷Cluster_7_H1_significant_lifespan_p90", etc.
meta = {"Region", "Slide"}
feat_cols = []
for c in agg.columns:
    if c in meta:
        continue
    # include if it starts with one of the prefixes and contains the expected pattern (H0/H1 + significant + stat)
    if any(c.startswith(f"{pref}_") for pref in prefixes) and (
        "_H0_significant_nBars" in c
        or "_H1_significant_nBars" in c
        or "_H0_significant_lifespan_mean" in c
        or "_H1_significant_lifespan_mean" in c
        or "_H0_significant_lifespan_std" in c
        or "_H1_significant_lifespan_std" in c
        or "H1_significant_birth_mean" in c
        or "H1_significant_birth_std" in c
    ):
        feat_cols.append(c)

if not feat_cols:
    raise RuntimeError("No feature columns matched the selected patterns; check naming in the trimmed CSVs.")

# 5) Drop zero-variance features (can't compute correlation)
nuniq = agg[feat_cols].nunique(dropna=False)
zero_var = nuniq[nuniq <= 1].index.tolist()
if zero_var:
    print("Dropping zero-variance features:", zero_var)
    feat_cols = [c for c in feat_cols if c not in zero_var]

# 6) Compute Spearman correlations for all unique unordered feature pairs
pairs = []
rhos = []
pvals = []
for i in range(len(feat_cols)):
    for j in range(i + 1, len(feat_cols)):
        f1 = feat_cols[i]
        f2 = feat_cols[j]
        # Spearman correlation (automatically handles ties)
        rho, p = spearmanr(agg[f1], agg[f2], nan_policy="omit")
        pairs.append((f1, f2))
        rhos.append(rho)
        pvals.append(p)

# 7) FDR correction
rej, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

# 8) Assemble result DataFrame
res = pd.DataFrame({
    "Feature1":       [a for a, _ in pairs],
    "Feature2":       [b for _, b in pairs],
    "Spearman_rho":   rhos,
    "rho_squared":    [rho**2 for rho in rhos],
    "p_value":        pvals,
    "p_value_FDR":    pvals_fdr,
    "Significant":    rej
})

# --- define thresholds ---
STRONG_R2 = 0.7        # what you used in the heatmap as a “strong” cutoff

total = len(res)

n_strong = (res["rho_squared"] >= STRONG_R2).sum()
pct_strong = n_strong / total * 100

print(f"Total tests = {total}")
print(f"Strong correlation (r² ≥ {STRONG_R2}): {n_strong}/{total} ({pct_strong:.1f}%)")


# 9) Summary & save
total = len(res)
sig = int(res["Significant"].sum())
print(f"Total tests={total}, significant after FDR={sig} ({sig/total:.2%})")
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
res.to_csv(OUT_CSV, index=False)
print("Saved global Spearman+FDR results → {OUT_CSV}")


# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# --- CONFIG ---
INPUT_CSV = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/TDA_NC/all_regions_slide_027stats_spearman_FDR.csv"
OUT_PNG = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/TDA_NC/correlations_heatmap_lower_triangle.png"
R2_THRESHOLD = 0.7  # toggle as needed
# ----------------

# 1. Load results
df = pd.read_csv(INPUT_CSV)

# 2. Build full list of features from Feature1/Feature2
features = sorted(set(df["Feature1"]).union(df["Feature2"]))
n = len(features)
feat_idx = {f: i for i, f in enumerate(features)}

# 3. Initialize matrices
r2_mat = np.full((n, n), np.nan)
sig_mat = np.zeros((n, n), dtype=bool)

# 4. Fill matrices (symmetrize)
for _, row in df.iterrows():
    i = feat_idx[row["Feature1"]]
    j = feat_idx[row["Feature2"]]
    rho2 = row["rho_squared"]
    r2_mat[i, j] = rho2
    r2_mat[j, i] = rho2
    sig = bool(row["Significant"])
    sig_mat[i, j] = sig
    sig_mat[j, i] = sig

# 5. Optionally cluster to get ordering (using distance = 1 - |rho| or based on r2)
# We'll cluster based on 1 - rho_squared (treating higher r2 = closer)
# Need a condensed distance matrix for linkage; fill diagonal with 0
d = 1.0 - r2_mat.copy()
np.fill_diagonal(d, 0.0)
# Replace any NaNs with max distance (1.0)
d = np.where(np.isnan(d), 1.0, d)
condensed = squareform(d, checks=False)
Z = linkage(condensed, method="average")
order = leaves_list(Z)
ordered_features = [features[i] for i in order]

# 6. Build plot matrix: lower triangle only, thresholded mask
to_plot = np.full((n, n), np.nan)
threshold_mask = (r2_mat >= R2_THRESHOLD).astype(int)  # for categorical coloring if desired

# Use the clustered order
idx_map = {f: k for k, f in enumerate(ordered_features)}
# Reorder matrices
r2_mat_ord = r2_mat[np.ix_(order, order)]
sig_mat_ord = sig_mat[np.ix_(order, order)]

# Prepare categorical matrix: red if > threshold else blue (only lower tri)
cat_mat = np.full((n, n), np.nan)
for i in range(n):
    for j in range(i + 1):
        if r2_mat_ord[i, j] >= R2_THRESHOLD:
            cat_mat[i, j] = 1  # high
        else:
            cat_mat[i, j] = 0  # low

# 7. Plot
fig, ax = plt.subplots(figsize=(22, 22))
cmap = mpl.colors.ListedColormap(['blue', 'red'])
bounds = [0, 0.5, 1]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

im = ax.imshow(cat_mat, cmap=cmap, norm=norm, interpolation='none')

# Overlay stars for significance (only lower triangle)
for i in range(n):
    for j in range(i + 1):
        if sig_mat_ord[i, j]:
            ax.text(j, i, '*', ha='center', va='center', fontsize=12, color='white')

# Tick labels
ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(ordered_features, rotation=90, fontsize=13)
ax.set_yticklabels(ordered_features, fontsize=13)
ax.set_xlim(-0.5, n - 0.5)
ax.set_ylim(n - 0.5, -0.5)

# Color legend for thresholded r2
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', edgecolor='none', label=f"$r^2$ ≥ {R2_THRESHOLD}"),
    Patch(facecolor='blue', edgecolor='none', label=f"$r^2$ < {R2_THRESHOLD}"),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='white', label='Significant (FDR)', markersize=10, linestyle='None')
]
ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=20)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
print(f"Saved heatmap to {OUT_PNG}")
plt.show()


# In[ ]:


# Step 5. Random Forest


# In[ ]:


import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score

# ---- CONFIG ----
ROOT = "/Volumes/My Passport/Spatial_Transcriptomics_data_final"
ROI_SUM_DIR = os.path.join(ROOT, "TDA_NC")
REGIONS = ["Healthy", "Peritumour", "Tumour"]
FEATURE_SUM_FILES = [
    os.path.join(ROI_SUM_DIR, reg, f"{reg}_all_ROIs_persistence_summary_clusters.csv")
    for reg in REGIONS
]
COMPARISONS = [
    ("Healthy", "Tumour"),
    ("Healthy", "Peritumour"),
    ("Peritumour", "Tumour"),
]

N_ESTIMATORS = 1000
RANDOM_STATE = 42
CV_SPLITS = 4

OUT_DIR = os.path.join(ROI_SUM_DIR, "rf_results")
os.makedirs(OUT_DIR, exist_ok=True)
# ----------------

def load_all_regions(files):
    dfs = []
    for path in files:
        region = os.path.basename(os.path.dirname(path))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Summary file not found: {path}")
        df = pd.read_csv(path)
        df["Region"] = region
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

if __name__ == "__main__":
    df_all = load_all_regions(FEATURE_SUM_FILES)
    # Grouping for CV: slide only (since slides span multiple regions)
    df_all["Group"] = df_all["Slide"].astype(str)

    for r1, r2 in COMPARISONS:
        df_pair = df_all[df_all["Region"].isin([r1, r2])].copy()

        # Feature columns: everything except metadata
        exclude_cols = {"Region", "Slide", "ROI", "Group"}
        features = [c for c in df_pair.columns if c not in exclude_cols]

        print(f"\n=== {r1} vs {r2} ===")
        print(f"ROIs used: {df_pair.shape[0]}, Features used: {len(features)}")

        X = df_pair[features].values.astype(np.float32)
        y = df_pair["Region"].values
        groups = df_pair["Group"].values  # slide-level grouping

        # Grouped CV
        gkf = GroupKFold(n_splits=CV_SPLITS)
        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        scores = cross_val_score(clf, X, y, cv=gkf, groups=groups, scoring="accuracy")
        print(f"Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

        # Per-fold feature importances (MDI)
        fold_imps = []
        for fold, (train_idx, _) in enumerate(gkf.split(X, y, groups), start=1):
            clf.fit(X[train_idx], y[train_idx])
            imps_fold = pd.Series(clf.feature_importances_, index=features, name=f"fold_{fold}")
            fold_imps.append(imps_fold)
        imp_df = pd.concat(fold_imps, axis=1)
        imp_df["mean_importance"] = imp_df.mean(axis=1)
        imp_df["std_importance"] = imp_df.std(axis=1)
        sorted_imp = imp_df.sort_values("mean_importance", ascending=False)

        # Save feature importances
        imp_csv = os.path.join(OUT_DIR, f"RF_{r1}_vs_{r2}_feature_mdi_folds.csv")
        sorted_imp.to_csv(imp_csv)
        print(f"Saved per-fold importances → {imp_csv}")

        # Save counts
        cnt_csv = os.path.join(OUT_DIR, f"RF_{r1}_vs_{r2}_feature_count.csv")
        pd.DataFrame({
            "Comparison": [f"{r1}_vs_{r2}"],
            "N_ROIs": [df_pair.shape[0]],
            "N_features": [len(features)]
        }).to_csv(cnt_csv, index=False)
        print(f"Saved counts → {cnt_csv}")


# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# ─── CONFIG ────────────────────────────────────────────────────────────────
ROOT         = "/Volumes/My Passport/Spatial_Transcriptomics_data_final"
RF_DIR       = os.path.join(ROOT, "TDA_NC", "rf_results")
OUT_PLOT_DIR = os.path.join(RF_DIR, "plots")
if not os.path.exists(OUT_PLOT_DIR):
    os.makedirs(OUT_PLOT_DIR)

# plotting parameters
top_n   = None   # if you want to limit number of features; else None
FIGSIZE = (12, 15)
PALETTE = [
    "#e6194b","#3cb44b","#ffe119","#1E90FF","#f58231","#911eb4",
    "#46f0f0","#bcf60c","#fac8e7","#008080","#e6beff","#9a6324",
    "#70360f","#f04d63","#aaffc3","#808000","#ffd8b1","#808080"
]
# ────────────────────────────────────────────────────────────────────────────

def get_prefix(feat):
    """Extract prefix before '_H0' or '_H1'."""
    for tok in ('_H0', '_H1'):
        if tok in feat:
            return feat.split(tok)[0]
    return feat

# only glob the exact CSVs written in step above
pattern = os.path.join(RF_DIR, "RF_*_feature_mdi_folds.csv")
files = sorted(glob.glob(pattern))

for path in files:
    # derive label from filename
    fname = os.path.basename(path)
    label = fname.replace('_feature_mdi_folds.csv', '')

    # read per-fold importances
    df = pd.read_csv(path, index_col=0)
    # df has columns fold_1, ..., mean_importance, std_importance

    # optionally limit to top_n
    imp = df['mean_importance']
    if top_n:
        imp = imp.nlargest(top_n)
    features = imp.index.tolist()

    means = df.loc[features, 'mean_importance']
    stds  = df.loc[features, 'std_importance']

    # assign colors by prefix
    prefixes = []
    for f in features:
        p = get_prefix(f)
        if p not in prefixes:
            prefixes.append(p)
    color_map = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(prefixes)}
    colors = [color_map[get_prefix(f)] for f in features]

    # plot
    plt.figure(figsize=FIGSIZE)
    y_pos = np.arange(len(features))
    plt.barh(y_pos, means[::-1], xerr=stds[::-1], color=colors[::-1], edgecolor='k')
    plt.yticks(y_pos, [features[i] for i in reversed(range(len(features)))])
    plt.xlabel('Mean Decrease Impurity')
    plt.margins(y=0.02)     # tighten the space above the top bar and below the bottom bar
    plt.title(label)
    plt.tight_layout()

    outpng = os.path.join(OUT_PLOT_DIR, f"{label}_mdi_barh.png")
    plt.savefig(outpng, dpi=250)
    plt.close()
    print(f"Saved plot: {outpng}")


# In[ ]:


# Step 6. Comparsions between any two regions for 80 features. (paired wilcoxon test + fdr P<0.05)


# In[ ]:


import os
import pandas as pd
import numpy as np

# ---- CONFIG ----
REGIONS = ["Tumour", "Peritumour", "Healthy"]
ROOT    = "/Volumes/My Passport/Spatial_Transcriptomics_data_final"
SLIDE_OUT  = os.path.join(ROOT, "TDA_NC", "slide_level_feature_summary.csv")
REGION_OUT = os.path.join(ROOT, "TDA_NC", "region_feature_roi_counts.csv")
TARGET_CLUSTERS = ["Cluster_0","Cluster_2","Cluster_7"]
# ----------------

slide_records = []
region_counts  = []

for region in REGIONS:
    summary_csv = os.path.join(
        ROOT, "TDA_NC", region,
        f"{region}_all_ROIs_persistence_summary_clusters.csv"
    )
    df = pd.read_csv(summary_csv).fillna(0.0)
    df['Region'] = region

    # (No ROI mask, just use all ROIs)

    # determine which features to include: single or pair among TARGET_CLUSTERS
    feat_cols = []
    for c in df.columns:
        if c in ('Region','Slide','ROI'):
            continue
        prefix = c.split('_H0')[0].split('_H1')[0]
        if '⟷' in prefix:
            a,b = prefix.split('⟷',1)
            if a in TARGET_CLUSTERS and b in TARGET_CLUSTERS:
                feat_cols.append(c)
        elif prefix in TARGET_CLUSTERS:
            feat_cols.append(c)

    # compute slide-level means & ROI counts: now for all ROIs
    for slide, sub in df.groupby('Slide'):
        idx = sub.index
        n_rois = len(idx)  # <- ALL ROIs for the slide
        for feat in feat_cols:
            mean_val = sub.loc[idx, feat].mean() if n_rois>0 else np.nan
            slide_records.append({
                'Region': region,
                'Slide': slide,
                'Feature': feat,
                'N_ROIs': n_rois,
                'Mean': mean_val
            })

# save slide-level summary
slide_df = pd.DataFrame.from_records(slide_records)
slide_df.to_csv(SLIDE_OUT, index=False)
print(f"Wrote slide-level summary {slide_df.shape} → {SLIDE_OUT}")

# region-level ROI counts (no change)
for (region, feat), grp in slide_df.groupby(['Region','Feature']):
    total = int(grp['N_ROIs'].sum())
    region_counts.append({
        'Region': region,
        'Feature': feat,
        'Total_N_ROIs': total
    })
region_df = pd.DataFrame.from_records(region_counts)
region_df.to_csv(REGION_OUT, index=False)
print(f"Wrote region-level counts {region_df.shape} → {REGION_OUT}")


# In[ ]:


# 0 2 7
import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# ---- CONFIG ----
ROOT                = "/Volumes/My Passport/Spatial_Transcriptomics_data_final"
SLIDE_SUM_CSV       = os.path.join(ROOT, "TDA_NC", "slide_level_feature_summary.csv")
OUT_CSV             = os.path.join(ROOT, "TDA_NC", "wilcoxon_feature_tests.csv")
FEATURE_COUNTS_CSV  = os.path.join(ROOT, "TDA_NC", "region_feature_slide_counts.csv")

REGIONS     = ["Tumour", "Peritumour", "Healthy"]
COMPARISONS = [
    ("Healthy",    "Tumour"),
    ("Healthy",    "Peritumour"),
    ("Peritumour", "Tumour")
]

ALPHA = 0.05

# Only keep immune‐related features (clusters 0,2,7 or their pairs)
TARGET_CLUSTERS = {"Cluster_0", "Cluster_2", "Cluster_7"}
def keep_feature(feat):
    prefix = feat.split("_H0")[0].split("_H1")[0]
    if "⟷" in prefix:
        a, b = prefix.split("⟷", 1)
        return (a in TARGET_CLUSTERS) and (b in TARGET_CLUSTERS)
    return prefix in TARGET_CLUSTERS

# 1) Load slide‑level summary
_df = pd.read_csv(SLIDE_SUM_CSV)

# 2) Pivot to wide format
pivot = _df.pivot_table(
    index=["Slide", "Feature"],
    columns="Region",
    values="Mean"
).reset_index()

# 3) Compute per‑feature slide counts
feature_counts = []
for feat in pivot["Feature"].unique():
    if not keep_feature(feat):
        continue
    feat_df = pivot[pivot["Feature"] == feat]
    for region in REGIONS:
        inc = feat_df[region].notna()
        feature_counts.append({
            "Feature": feat,
            "Region": region,
            "N_Slides": int(inc.sum()),
            "Excluded_Slides": ";".join(feat_df.loc[~inc, "Slide"].unique())
        })
counts_df = pd.DataFrame(feature_counts)
counts_df.to_csv(FEATURE_COUNTS_CSV, index=False)
print(f"Wrote slide counts → {FEATURE_COUNTS_CSV}")

# 4) Paired Wilcoxon tests
results = []
for r1, r2 in COMPARISONS:
    comp = pivot.dropna(subset=[r1, r2])
    for feat in comp["Feature"].unique():
        if not keep_feature(feat):
            continue
        sub = comp[comp["Feature"] == feat]
        x = sub[r1].values
        y = sub[r2].values
        n = len(x)
        if n > 0:
            try:
                stat, p = wilcoxon(x, y)
            except ValueError:
                stat, p = np.nan, np.nan
        else:
            stat, p = np.nan, np.nan
        results.append({
            "Feature": feat,
            "Region_1": r1,
            "Region_2": r2,
            "N_Slides": n,
            "Statistic": stat,
            "P_value": p
        })

# 5) FDR correction
out_df = pd.DataFrame(results)
corrected = []
for (r1, r2), grp in out_df.groupby(["Region_1", "Region_2"]):
    rej, p_adj, _, _ = multipletests(grp["P_value"], alpha=ALPHA, method="fdr_bh")
    grp = grp.assign(P_adj=p_adj, Significant=rej)
    corrected.append(grp)
final = pd.concat(corrected, ignore_index=True).sort_values(["Region_1","Region_2","P_adj"])

# 6) Save results
final.to_csv(OUT_CSV, index=False)
print(f"Wrote Wilcoxon results → {OUT_CSV}")


# In[ ]:


# p-adj version box plots
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─── CONFIG ────────────────────────────────────────────────────────────────
ROOT           = "/Volumes/My Passport/Spatial_Transcriptomics_data_final"
SLIDE_SUM_CSV  = os.path.join(ROOT, "TDA_NC", "slide_level_feature_summary.csv")
FDR_CSV        = os.path.join(ROOT, "TDA_NC", "wilcoxon_feature_tests.csv")
OUTPUT_DIR     = os.path.join(ROOT, "TDA_NC", "boxplots_per_feature")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_REGIONS = ["Healthy", "Peritumour", "Tumour"]
COLORS      = {"Healthy":"lightgreen", "Peritumour":"skyblue", "Tumour":"orange"}
PAIRS       = [("Healthy","Peritumour"), ("Healthy","Tumour"), ("Peritumour","Tumour")]

# load data
df_means = pd.read_csv(SLIDE_SUM_CSV)
fdr       = pd.read_csv(FDR_CSV)

# get only the features we want
all_feats = sorted(df_means["Feature"].unique())
feats     = [f for f in all_feats if keep_feature(f)]

for feat in feats:
    present = {r: not df_means[(df_means.Feature==feat)&(df_means.Region==r)].Mean.dropna().empty
               for r in ALL_REGIONS}
    regions = [r for r in ALL_REGIONS if present[r]]

    data, slide_ids, bounds = [], [], []
    for r in regions:
        sub = df_means[(df_means.Feature==feat)&(df_means.Region==r)].dropna(subset=["Mean"])
        vals = sub["Mean"].values
        sids = sub["Slide"].str.split("_").str[-1].values
        data.append(vals); slide_ids.append(sids)
        if len(vals):
            q1, q3 = np.percentile(vals,[25,75])
            iqr     = q3 - q1
            lb = q1-1.5*iqr; ub = q3+1.5*iqr
        else:
            lb = ub = np.nan
        bounds.append((lb,ub))

    if all(len(d)==0 for d in data):
        continue

    fig, ax = plt.subplots(figsize=(6,7))
    x = np.arange(len(regions))

    bp = ax.boxplot(data, positions=x, widths=0.6,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(color='k'),
                    whiskerprops=dict(color='k'),
                    capprops=dict(color='k'),
                    medianprops=dict(color='k'))
    for patch, r in zip(bp['boxes'], regions):
        patch.set_facecolor(COLORS[r]); patch.set_alpha(0.5)

    for i, arr in enumerate(data):
        if len(arr)==0: continue
        jitter = np.random.normal(0,0.03,size=len(arr))
        ax.scatter(i+jitter, arr, color='k', s=9, alpha=0.7, zorder=3)
        lb, ub = bounds[i]
        for j, val in enumerate(arr):
            if not np.isnan(lb) and (val<lb or val>ub):
                off = 0.02*(ub-lb) if ub>lb else 0.1
                ax.text(i+jitter[j], val+off, slide_ids[i][j],
                        fontsize=6, ha='center', va='bottom', color='k')

    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=45, ha='right')
    ax.set_ylabel("Mean persistence feature value")
    ax.set_title(feat)

    # ─── add adjusted‐p labels instead of stars ─────────────────────────────
    for r1, r2 in PAIRS:
        if r1 in regions and r2 in regions:
            row = fdr[(fdr.Region_1==r1)&(fdr.Region_2==r2)&(fdr.Feature==feat)]
            if not row.empty and row.Significant.iloc[0]:
                # pull the adjusted‐p value
                p_adj = row["P_adj"].iloc[0]
                label = f"p-adj={p_adj:.3f}"
                i1, i2 = regions.index(r1), regions.index(r2)
                all_vals = np.concatenate(data)
                top, bot = all_vals.max(), all_vals.min()
                h  = (top - bot)*0.01 if top>bot else 0.1
                y0 = top + 2*h; y1 = top + 3*h
                ax.plot([i1,i1,i2,i2],[y0,y1,y1,y0],color='k',lw=1)
                ax.text((i1+i2)/2, y1+0.02*h, label,
                        ha='center', va='bottom', fontsize=7)
    # ────────────────────────────────────────────────────────────────────────

    plt.tight_layout()
    fn = feat.replace('⟷','_').replace(' ','_').replace('/','_')+".png"
    fig.savefig(os.path.join(OUTPUT_DIR,fn),dpi=250)
    plt.close(fig)

    print(f"Saved boxplot for {feat} → {fn}")


# In[ ]:


# New Volcano plots 0 4 5 7
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ─── CONFIG ────────────────────────────────────────────────────────────
ROOT        = "/Volumes/My Passport/Spatial_Transcriptomics_data_final"
SLIDE_MEANS = os.path.join(ROOT, "TDA_NC", "slide_level_feature_summary.csv")
TESTS_CSV   = os.path.join(ROOT, "TDA_NC", "wilcoxon_feature_tests.csv")
OUTDIR      = os.path.join(ROOT, "TDA_NC", "volcano_plots_TDA")
os.makedirs(OUTDIR, exist_ok=True)

REGION_PAIRS = [
    ("Healthy",    "Peritumour"),
    ("Healthy",    "Tumour"),
    ("Peritumour", "Tumour")
]

# Only immune clusters (0,2,7) or pairs among them
IMMUNE_CLUSTERS = {"Cluster_0","Cluster_2","Cluster_7"}

def is_immune_feature(feat):
    pref = feat.split("_H0")[0].split("_H1")[0]
    if "⟷" in pref:
        a,b = pref.split("⟷",1)
        return (a in IMMUNE_CLUSTERS) and (b in IMMUNE_CLUSTERS)
    return pref in IMMUNE_CLUSTERS

# ─── LOAD DATA ─────────────────────────────────────────────────────────────
if not os.path.isfile(SLIDE_MEANS) or not os.path.isfile(TESTS_CSV):
    sys.exit("Missing input CSVs")
slides = pd.read_csv(SLIDE_MEANS)
tests  = pd.read_csv(TESTS_CSV)

# ─── PREPARE MEDIANS ───────────────────────────────────────────────────────
long = slides[['Slide','Region','Feature','Mean']]
med = ( long.groupby(['Feature','Region'])['Mean']
           .median()
           .reset_index() )
med = med.pivot(index='Feature', columns='Region', values='Mean')

# ─── COLOR MAP BY PREFIX ───────────────────────────────────────────────────
all_feats = sorted(med.index)
prefs = []
for f in all_feats:
    p = f.split("_H0")[0].split("_H1")[0]
    if p not in prefs:
        prefs.append(p)
PALETTE = [
    "#e6194b","#3cb44b","#ffe119","#1E90FF","#f58231","#911eb4",
    "#46f0f0","#bcf60c","#fac8e7","#008080","#e6beff","#9a6324",
    "#70360f","#f04d63","#aaffc3","#808000","#ffd8b1","#808080"
]
color_map = {p: PALETTE[i % len(PALETTE)] for i,p in enumerate(prefs)}
feat_color = {f: color_map[f.split("_H0")[0].split("_H1")[0]] for f in all_feats}

# ─── VOLCANO PLOTS ────────────────────────────────────────────────────────
for r1, r2 in REGION_PAIRS:
    fout = os.path.join(OUTDIR, f"volcano_{r1}_vs_{r2}.png")
    fig, ax = plt.subplots(figsize=(25,12))

    neg_feats, pos_feats = [], []
    deltas = {}

    for f in all_feats:
        if not is_immune_feature(f):
            continue
        if f not in med.index:
            continue
        v1 = med.loc[f, r1]
        v2 = med.loc[f, r2]
        if np.isnan(v1) or np.isnan(v2):
            continue
        d = v2 - v1
        row = tests[(tests.Region_1==r1)&(tests.Region_2==r2)&(tests.Feature==f)]
        if row.empty:
            continue
        p_adj = row.P_adj.values[0]
        neglogp = -np.log10(p_adj if p_adj>0 else 1e-300)

        ax.scatter(
            d, neglogp,
            color=feat_color[f],
            edgecolors='k', linewidths=0.8,
            s=50, zorder=2
        )

        # Always collect for legend, regardless of significance
        deltas[f] = d
        (neg_feats if d < 0 else pos_feats).append(f)

    ax.axhline(-np.log10(0.05), ls='--', c='k', lw=1)
    ax.set_xlabel(f"{r2} median – {r1} median")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title(f"Volcano: {r1} vs {r2}")

    # Sort for legend: neg_feats (lowest to highest), pos_feats (lowest to highest)
    neg_feats = sorted(set(neg_feats), key=lambda x: deltas[x])
    pos_feats = sorted(set(pos_feats), key=lambda x: deltas[x])

    neg_handles = [Line2D([0],[0], marker='o', color=feat_color[f], markeredgecolor='k', linestyle='') for f in neg_feats]
    pos_handles = [Line2D([0],[0], marker='o', color=feat_color[f], markeredgecolor='k', linestyle='') for f in pos_feats]

    plt.subplots_adjust(right=0.7)
    # Legend is ALWAYS shown, even if list is empty
    leg1 = ax.legend(neg_handles, neg_feats, title=r1,
                     loc='upper left', bbox_to_anchor=(1,1), frameon=False)
    ax.add_artist(leg1)
    ax.legend(pos_handles, pos_feats, title=r2,
              loc='upper left', bbox_to_anchor=(1.4,1), frameon=False)

    plt.tight_layout(rect=[0,0,0.88,1])
    fig.savefig(fout, dpi=300)
    plt.close(fig)
    print(f"Saved {fout}")


# In[ ]:


# PCA and UMAP


# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import scanpy as sc  # for ComBat

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
ROOT       = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/TDA_NC"
REGIONS    = ["Tumour", "Peritumour", "Healthy"]
SUMMARY_FN = "{region}_all_ROIs_persistence_summary_clusters.csv"

OUT_DIR = os.path.join(ROOT, "027_TDAs_PCA_UMAP_combat")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 1) READ, ANNOTATE & CONCATENATE ──────────────────────────────────────────
dfs = []
for region in REGIONS:
    in_path = os.path.join(ROOT, region, SUMMARY_FN.format(region=region))
    df = pd.read_csv(in_path).fillna(0.0)
    df["Region"] = region
    keep = ["Region", "Slide", "ROI"] + [c for c in df.columns if c not in ("Region", "Slide", "ROI")]
    df_sub = df[keep]
    out_csv = os.path.join(OUT_DIR, f"{region}_all_ROIs_TDA.csv")
    df_sub.to_csv(out_csv, index=False)
    dfs.append(df_sub)

big_df = pd.concat(dfs, ignore_index=True).fillna(0.0)
big_df_idx = big_df.set_index(["Region", "Slide", "ROI"])

# ─── 2) FEATURE COLUMNS (keep all except metadata) ───────────────────────────
feature_cols = [c for c in big_df_idx.columns if c not in ["Region", "Slide", "ROI"]]

# ─── 3) SCALE & COMBAT ───────────────────────────────────────────────────────
X = big_df_idx[feature_cols].values
Xs = StandardScaler().fit_transform(X)

# Build AnnData for ComBat
adata = sc.AnnData(Xs.astype(np.float32))
# add batch (Slide)
adata.obs['batch'] = big_df_idx.index.get_level_values('Slide').astype(str).values
# one-hot encode region covariate, drop first level
region_dummies = pd.get_dummies(
    big_df_idx.index.get_level_values('Region'), prefix='region', drop_first=True
)
for col in region_dummies.columns:
    adata.obs[col] = region_dummies[col].values.astype(np.float32)
# run ComBat to remove batch effects while preserving Region
sc.pp.combat(adata, key='batch', covariates=region_dummies.columns.tolist())
# corrected data
Xc = adata.X

# ─── 4) PCA ─────────────────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=0)
coords = pca.fit_transform(Xc)
loadings = pd.DataFrame(pca.components_.T,
                        index=feature_cols,
                        columns=["PC1", "PC2"])
explained = pd.Series(pca.explained_variance_ratio_,
                      index=["PC1", "PC2"],
                      name="explained_variance_ratio")
coords_df = pd.DataFrame(coords,
                         columns=["PC1", "PC2"],
                         index=big_df_idx.index)
coords_df.reset_index().to_csv(os.path.join(OUT_DIR, "PCA_coords.csv"), index=False)
loadings.to_csv(os.path.join(OUT_DIR, "PCA_loadings.csv"))
explained.to_csv(os.path.join(OUT_DIR, "PCA_explained_variance.csv"))

# ─── 5) Plot PCA by Region ─────────────────────────────────────────────────
color_map = {"Tumour": "orange", "Peritumour": "lightblue", "Healthy": "lightgreen"}
plt.figure(figsize=(7, 6))
for region in REGIONS:
    mask = coords_df.index.get_level_values("Region") == region
    plt.scatter(coords[mask, 0], coords[mask, 1],
                label=region, s=12, alpha=0.7, color=color_map[region])
handles, labels = plt.gca().get_legend_handles_labels()
order = [labels.index(r) for r in REGIONS]
plt.legend([handles[i] for i in order],
           [labels[i] for i in order],
           bbox_to_anchor=(1.02, 1), loc="upper left",
           frameon=False, title="Region")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("PCA (Combat-corrected, all TDA features)")
plt.tight_layout(rect=[0.8, 0.8, 1, 1])
plt.savefig(os.path.join(OUT_DIR, "PCA_plot_by_region.png"), dpi=300, bbox_inches="tight")
plt.close()

# ─── 6) Plot PCA by Slide ─────────────────────────────────────────────────
plt.figure(figsize=(7, 6))
slides = coords_df.index.get_level_values("Slide").unique()
slide_list = list(slides)
colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(slide_list)))
for i, slide in enumerate(slide_list):
    mask = coords_df.index.get_level_values("Slide") == slide
    plt.scatter(coords[mask, 0], coords[mask, 1],
                label=slide, s=12, alpha=0.7, color=colors[i])
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, title="Slide", fontsize=8)
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("PCA (Combat-corrected, by Slide)")
plt.tight_layout(rect=[0.8, 0.8, 1, 1])
plt.savefig(os.path.join(OUT_DIR, "PCA_plot_by_slide.png"), dpi=300, bbox_inches="tight")
plt.close()

# ─── 7) UMAP ────────────────────────────────────────────────────────────────
reducer = umap.UMAP(n_neighbors=50, min_dist=0.8, spread=2, random_state=42)
Z = reducer.fit_transform(Xc)
umap_df = pd.DataFrame(Z, columns=["UMAP1", "UMAP2"], index=big_df_idx.index)
umap_df.reset_index().to_csv(os.path.join(OUT_DIR, "UMAP_coords.csv"), index=False)

# ─── 8) Plot UMAP by Region ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
for region in REGIONS:
    mask = umap_df.index.get_level_values("Region") == region
    ax.scatter(Z[mask, 0], Z[mask, 1],
               label=region, s=12, alpha=0.7,
               edgecolors="none", color=color_map[region])
handles, labels = ax.get_legend_handles_labels()
order = [labels.index(r) for r in REGIONS]
ax.legend([handles[i] for i in order],
          [labels[i] for i in order],
          bbox_to_anchor=(1.02, 1), loc="upper left",
          frameon=False, title="Region")
ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
ax.set_title("UMAP (Combat-corrected, all TDA features)")
plt.tight_layout(rect=[0.8, 0.8, 1, 1])
fig.savefig(os.path.join(OUT_DIR, "UMAP_plot_by_region.png"), dpi=300, bbox_inches="tight")
plt.close()

# ─── 9) Plot UMAP by Slide ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
for i, slide in enumerate(slide_list):
    mask = umap_df.index.get_level_values("Slide") == slide
    ax.scatter(Z[mask, 0], Z[mask, 1],
               label=slide, s=12, alpha=0.7, edgecolors="none", color=colors[i])
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, title="Slide", fontsize=8)
ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
ax.set_title("UMAP (Combat-corrected, by Slide)")
plt.tight_layout(rect=[0.8, 0.8, 1, 1])
fig.savefig(os.path.join(OUT_DIR, "UMAP_plot_by_slide.png"), dpi=300, bbox_inches="tight")
plt.close()

print("Done.")


# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import scanpy as sc  # for ComBat

# ---- CONFIG ----
ROOT = "/Volumes/My Passport/Spatial_Transcriptomics_data_final/TDA_NC"
OUT_DIR = os.path.join(ROOT, "rf_top20featuregroup_PCA_UMAP_combat")
os.makedirs(OUT_DIR, exist_ok=True)

REGION_PAIRS = [
    ("Peritumour", "Tumour"),
    ("Healthy", "Peritumour"),
    ("Healthy", "Tumour"),
]
SUMMARY_FN = "{region}_all_ROIs_persistence_summary_clusters.csv"

region_colors = {
    "Tumour": "orange",
    "Peritumour": "lightblue",
    "Healthy": "lightgreen",
}

# Load all region ROI summary data
dfs = []
for region in {r for pair in REGION_PAIRS for r in pair}:
    path = os.path.join(ROOT, region, SUMMARY_FN.format(region=region))
    df = pd.read_csv(path).fillna(0.0)
    df["Region"] = region
    dfs.append(df)
full_df = pd.concat(dfs, ignore_index=True).set_index(["Region", "Slide", "ROI"])

for region1, region2 in REGION_PAIRS:
    print(f"\n=== {region1} vs {region2} ===")
    # 1) Load RF importances & pick top 20
    rf_csv = os.path.join(ROOT, "rf_results", f"RF_{region1}_vs_{region2}_feature_mdi_folds.csv")
    imp = pd.read_csv(rf_csv, index_col=0).sort_values("mean_importance", ascending=False)
    top_feats = imp.index[:20].tolist()

    # 2) Subset to those two regions (NO ROI FILTERING)
    mask = full_df.index.get_level_values("Region").isin([region1, region2])
    sub_df = full_df[mask]

    # 3) Build feature matrix (all ROIs, top 20 features)
    feats = [f for f in top_feats if f in sub_df.columns]
    X = sub_df[feats].values
    df_meta = sub_df.reset_index()[["Slide", "Region"]].rename(columns={"Slide": "slide_ID", "Region": "region"})
    slides = df_meta["slide_ID"].astype(str).values
    regions = df_meta["region"].values

    # 4) Standardize
    Xs = StandardScaler().fit_transform(X)

    # 5) Build AnnData for ComBat
    X_nb     = Xs.astype(np.float32)
    adata_nb = sc.AnnData(X_nb)
    adata_nb.obs["batch"] = slides
    # one‐hot encode region, drop first level to avoid collinearity
    region_dummies = pd.get_dummies(df_meta["region"], prefix="region", drop_first=True)
    for col in region_dummies.columns:
        adata_nb.obs[col] = region_dummies[col].astype(np.float32).values

    # 6) Run ComBat with slide batch and region covariates
    sc.pp.combat(
        adata_nb,
        key="batch",
        covariates=region_dummies.columns.tolist()
    )
    Xc = adata_nb.X  # corrected data

    # 7) PCA on batch‐corrected data
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(Xc)
    pcs_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
    pcs_df["Slide"]  = slides
    pcs_df["Region"] = regions

    slide_list = np.unique(slides)
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, len(slide_list)))

    # PCA colored by Slide
    plt.figure(figsize=(7,6))
    for i, sl in enumerate(slide_list):
        m = pcs_df["Slide"] == sl
        plt.scatter(pcs_df.loc[m, "PC1"], pcs_df.loc[m, "PC2"],
                    label=sl, s=14, alpha=0.7, color=colors[i])
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"PCA: {region1} vs {region2} [top20 RF features, ComBat]")
    plt.legend(bbox_to_anchor=(1.02,1), loc="upper left",
               frameon=False, title="Slide", fontsize=8)
    plt.tight_layout(rect=[0.8,0.8,1,1])
    plt.savefig(os.path.join(OUT_DIR, f"PCA20_ComBat_{region1}_vs_{region2}_bySlide.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # PCA colored by Region
    plt.figure(figsize=(7,6))
    for reg in [region1, region2]:
        m = pcs_df["Region"] == reg
        plt.scatter(pcs_df.loc[m, "PC1"], pcs_df.loc[m, "PC2"],
                    label=reg, s=14, alpha=0.7, color=region_colors[reg])
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"PCA: {region1} vs {region2} [top20 RF features, ComBat]")
    plt.legend(bbox_to_anchor=(1.02,1), loc="upper left",
               frameon=False, title="Region")
    plt.tight_layout(rect=[0.8,0.8,1,1])
    plt.savefig(os.path.join(OUT_DIR, f"PCA20_ComBat_{region1}_vs_{region2}_byRegion.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # 8) UMAP on batch‐corrected data
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.8, spread=2, random_state=42)
    ums = reducer.fit_transform(Xc)
    ums_df = pd.DataFrame(ums, columns=["UMAP1", "UMAP2"])
    ums_df["Slide"]  = slides
    ums_df["Region"] = regions

    # UMAP colored by Slide
    plt.figure(figsize=(7,6))
    for i, sl in enumerate(slide_list):
        m = ums_df["Slide"] == sl
        plt.scatter(ums_df.loc[m, "UMAP1"], ums_df.loc[m, "UMAP2"],
                    label=sl, s=14, alpha=0.7, edgecolors="none", color=colors[i])
    plt.xlabel("UMAP1"); plt.ylabel("UMAP2")
    plt.title(f"UMAP: {region1} vs {region2} [top20 RF features, ComBat]")
    plt.legend(bbox_to_anchor=(1.02,1), loc="upper left",
               frameon=False, title="Slide", fontsize=8)
    plt.tight_layout(rect=[0.8,0.8,1,1])
    plt.savefig(os.path.join(OUT_DIR, f"UMAP20_ComBat_{region1}_vs_{region2}_bySlide.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # UMAP colored by Region
    plt.figure(figsize=(7,6))
    for reg in [region1, region2]:
        m = ums_df["Region"] == reg
        plt.scatter(ums_df.loc[m, "UMAP1"], ums_df.loc[m, "UMAP2"],
                    label=reg, s=14, alpha=0.7, edgecolors="none", color=region_colors[reg])
    plt.xlabel("UMAP1"); plt.ylabel("UMAP2")
    plt.title(f"UMAP: {region1} vs {region2} [top20 RF features, ComBat]")
    plt.legend(bbox_to_anchor=(1.02,1), loc="upper left",
               frameon=False, title="Region")
    plt.tight_layout(rect=[0.8,0.8,1,1])
    plt.savefig(os.path.join(OUT_DIR, f"UMAP20_ComBat_{region1}_vs_{region2}_byRegion.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved ComBat‐corrected PCA & UMAP for {region1} vs {region2}")

print("All done!")


# In[ ]:




