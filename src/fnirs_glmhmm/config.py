"""Dataset locations and a few constants that get reused everywhere."""

import os
from pathlib import Path

DATASET_ROOT = Path(os.environ.get("GRADCPT_ROOT", "/projectnb/nphfnirs/s/datasets/gradCPT_NN24"))

# Output of the cedalion pipeline: one pickle per subject per preprocessing variant.
DERIV_ROOT = DATASET_ROOT / "derivatives/cedalion/pipeline_reorder/processed_data"

# Several variants sit side by side in each subject folder; this is the adot-probe,
# lR-1e-5, v26 one. 21 of the 23 subject folders have it.
PARCEL_FILE_TEMPLATE = (
    "{sub}_task-gradCPT_adot-probe_spatialdim-vertex_IR_ts_{variant}_lR-1e-5_v26.pkl"
)

# Preprocessing notes to confirm with Laura: ols used TDDR and a 0.5 Hz lowpass
# before parcel projection; ar_irls used raw input.
VARIANTS = ("ols", "ar_irls")

# Non-cortical entries excluded from parcel analyses.
NON_CORTICAL_PARCELS = ("Background+FreeSurfer_Defined_Medial_Wall", "scalp")

# Where analysis output goes. Both are gitignored.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(os.environ.get("FNIRS_RESULTS", PROJECT_ROOT / "results"))
FIGURES_DIR = Path(os.environ.get("FNIRS_FIGURES", PROJECT_ROOT / "figures"))
