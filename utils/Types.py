from enum import Enum

class SegMode(Enum):
    sphere = 'sphere'
    gauss = 'gauss'

class SNR(Enum):
    TYPE_1 = 'snr_1'
    TYPE_2 = 'snr_2'
    TYPE_4 = 'snr_4'
    TYPE_7 = 'snr_7'

class Density(Enum):
    HIGH = 'density_high'
    MID = 'density_mid'
    LOW = 'density_low'