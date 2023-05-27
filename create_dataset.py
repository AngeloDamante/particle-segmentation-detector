import logging
from utils.logger import configure_logger

configure_logger(logging.INFO)


def split_dataset(perc: int):
    """Split dataset into training, testing, validation directories

        Input percentage  is the value for training. The rest of the percentage
        is divided equally between testing and validation.

    :param perc: percentage of split for training
    :return:
    """
    # TODO
    pass


def delete_folder():
    # TODO
    pass


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
