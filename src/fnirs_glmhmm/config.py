"""Dataset locations and a few constants that get reused everywhere."""

import os
from pathlib import Path

DATASET_ROOT = Path(os.environ.get("GRADCPT_ROOT", "/projectnb/nphfnirs/s/datasets/gradCPT_NN24"))

# Output of the cedalion pipeline: one pickle per subject per preprocessing variant.
DERIV_ROOT = DATASET_ROOT / "derivatives/cedalion/pipeline_reorder/processed_data"

PARCEL_FILE_TEMPLATE = "{sub}_adot-probe_spatialdim-vertex_IR_ts_{variant}_IR-1e-5_v26.pkl"

# ols was TDDR + 0.5 Hz lowpass before projection to parcel space; ar_irls was raw.
# Laura can confirm this
VARIANTS = ("ols", "ar_irls")

# Parcels that come along for the ride but are not cortex.
NON_CORTICAL_PARCELS = ("Background+FreeSurfer_Defined_Medial_Wall", "scalp")

# Where analysis output goes. Both are gitignored.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(os.environ.get("FNIRS_RESULTS", PROJECT_ROOT / "results"))
FIGURES_DIR = Path(os.environ.get("FNIRS_FIGURES", PROJECT_ROOT / "figures"))
