import logging
import os
import shutil
from utils.definitions import DTS_DIR, DTS_TRAIN_PATH, DTS_TEST_PATH, DTS_VALIDATION_PATH, DTS_RAW_PATH,TIME_INTERVAL
from utils.Types import SNR, Density
from utils.logger import configure_logger
from utils.compute_path import get_data_path

configure_logger(logging.INFO)


def split_dataset(snr: SNR, density: Density, p_train: int=80):
    """Split dataset into training, testing, validation directories

        Input percentage  is the value for training. The rest of the percentage
        is divided equally between testing and validation.

    :param p_train: percentage of splitting for training
    :param density:
    :param snr:
    :return:
    """
    p_test = (TIME_INTERVAL - p_train) / 2
    p_val = p_test
    os.makedirs(DTS_TRAIN_PATH, exist_ok=True)
    os.makedirs(DTS_TEST_PATH, exist_ok=True)
    os.makedirs(DTS_VALIDATION_PATH, exist_ok=True)

    for time in range(p_train):
        path = get_data_path(snr.value, density.value, t=time, is_npz=True, root=DTS_RAW_PATH)
        shutil.copy2(path, DTS_TRAIN_PATH)

    for time in range(p_test):
        t = p_train + time
        path = get_data_path(snr.value, density.value, t=t, is_npz=True, root=DTS_RAW_PATH)
        shutil.copy2(path, DTS_TEST_PATH)

    for time in range(p_val):
        t = p_train + p_test + time
        path = get_data_path(snr.value, density.value, t=t, is_npz=True, root=DTS_RAW_PATH)
        shutil.copy2(path, DTS_VALIDATION_PATH)


def delete_folder(folder: str):
    path_folder = os.path.join(DTS_DIR, folder)
    for filename in os.listdir(path_folder):
        file_path = os.path.join(path_folder, filename)
        os.remove(file_path)


def save_slices():
    # TODO
    pass


def json_parser():
    # TODO
    pass

# def _save_data(self, img_3d: np.ndarray, time: int, directory: str, save_img: bool):
#     """ Save data
#
#     :param img_3d:
#     :param time:
#     :param directory:
#     :param save_img:
#     :return:
#     """
#     # create dir data
#     dir_data = os.path.join(DTS_SEG_DATA, directory)
#     os.makedirs(dir_data, exist_ok=True)
#
#     # save npy
#     np.save(os.path.join(dir_data, f't_{str(time).zfill(3)}'), img_3d)
#
#     # create img dir
#     if not save_img: return
#     dir_img = os.path.join(DTS_SEG_IMG, directory)
#     os.makedirs(dir_img, exist_ok=True)
#
#     # save slices
#     for depth in range(self.size[2]):
#         img_name = f't_{str(time).zfill(3)}_z_{str(depth).zfill(2)}.tiff'
#         img = Image.fromarray(img_3d[:, :, depth].astype(np.uint8))
#         img.save(os.path.join(dir_img, img_name))
