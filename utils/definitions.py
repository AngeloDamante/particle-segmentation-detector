""" Main Definitions for project """

import os
from pathlib import Path

# main dirs
ROOT_DIR = Path(__file__).absolute().parent.parent
DTS_DIR = os.path.join(ROOT_DIR, "Dataset")

# Original Dataset
DTS_CHALLANGE = os.path.join(DTS_DIR, "Challenge")
DTS_VIRUS = os.path.join(DTS_CHALLANGE, "VIRUS")
DTS_GTH = os.path.join(DTS_CHALLANGE, "ground_truth")

# Segmented Datasets
DTS_SEG_1 = os.path.join(DTS_DIR, "seg_technique_1")