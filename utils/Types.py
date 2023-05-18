"""Define utils Types to facilitate extraction"""

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


# dict to parsing
mapSegMode = {'sphere': SegMode.sphere, 'gauss': SegMode.gauss}
mapSNR = {'snr_1': SNR.TYPE_1, 'snr_2': SNR.TYPE_2, 'snr_4': SNR.TYPE_4, 'snr_7': SNR.TYPE_7}
mapDensity = {'density_high': Density.HIGH, 'density_mid': Density.MID, 'density_low': Density.LOW}
