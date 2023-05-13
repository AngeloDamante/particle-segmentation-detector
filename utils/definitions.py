""" Main Definitions for project """

import os
from pathlib import Path

# main dirs
ROOT_DIR = Path(__file__).absolute().parent.parent
DTS_DIR = os.path.join(ROOT_DIR, "Dataset")

# Original Dataset
DTS_CHALLENGE = os.path.join(DTS_DIR, "Challenge")
DTS_VIRUS = os.path.join(DTS_CHALLENGE, "VIRUS")
DTS_GTH = os.path.join(DTS_CHALLENGE, "ground_truth")