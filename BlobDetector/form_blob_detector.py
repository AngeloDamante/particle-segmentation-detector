"""UI for Blob Detector"""
import logging
import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtGui, QtWidgets
from utils.compute_path import get_data_path
from models.unet import UNET
from models.vit import SegFormer
from utils.definitions import (
    EXPERIMENTS_PATH,
    DTS_RAW_PATH,
    DEPTH,
    TIME_INTERVAL,
    mapSNR,
    mapDensity,
)


class Ui_MainWindow(object):
    def __init__(self):
        # objs
        self.model = nn.Module()
        self.blob_detector_params = params = cv2.SimpleBlobDetector_Params()
        self.input_image = None  # (H,W,D)
        self.output_image = None  # (H,W,D)

        # flags
        self.blob_activate = False
        self.model_loaded = False

    def load_results(self):
        """Load x and y

        :return:
        """
        snr = mapSNR[self.snr_comboBox.currentText()]
        density = mapDensity[self.density_comboBox.currentText()]
        t = int(self.t_comboBox.currentText())
        data = np.load(get_data_path(snr, density, t, is_npz=True, root=DTS_RAW_PATH))
        x = data['img']
        _x = np.divide(x, 255.0, dtype=np.float32)
        _x = torch.from_numpy(_x)
        _x = torch.permute(_x, (2, 0, 1)).unsqueeze(0)
        with torch.no_grad():
            y = torch.sigmoid(self.model(_x))
            y = torch.permute(y.squeeze(0), (1, 2, 0)).numpy() * 255
        self.input_image = data['target']
        self.output_image = y.astype(np.uint8)

    def print_image(self, image, is_3d=True):
        """Print Images

        :return:
        """
        if is_3d:
            z = int(self.z_comboBox.currentText())
            frame_in = cv2.cvtColor(image[:, :, z], cv2.COLOR_GRAY2RGB)
        else:
            frame_in = image

        # slice input
        h, w = image.shape[:2]
        bytesPerLine = 3 * w
        qImage = QImage(frame_in.data, w, h, bytesPerLine, QImage.Format.Format_RGB888)
        pixmap = QPixmap(qImage)
        pixmap = pixmap.scaled(512, 512, Qt.KeepAspectRatio)
        return pixmap

    def set_blobber_params(self):
        """Set blobber params and active blob detector

        :return:
        """
        if not self.blob_activate: return
        print("[ BLOB DETECTOR SETTED ]")
        self.blob_detector_params.minThreshold = int(self.blob_min_threshold.text())
        self.blob_detector_params.maxThreshold = int(self.blob_max_threshold.text())

        self.blob_detector_params.filterByCircularity = self.blob_filter_circularity_box.isChecked()
        if self.blob_filter_circularity_box.isChecked():
            self.blob_detector_params.minCircularity = float(self.blob_min_circularity.text())
            self.blob_detector_params.maxCircularity = float(self.blob_max_circularity.text())

        self.blob_detector_params.filterByConvexity = self.blob_filter_convexity_box.isChecked()
        if self.blob_filter_convexity_box.isChecked():
            self.blob_detector_params.minConvexity = float(self.blob_min_convexity.text())
            self.blob_detector_params.maxConvexity = float(self.blob_max_convexity.text())

        self.blob_detector_params.filterByArea = self.blob_filter_area_box.isChecked()
        if self.blob_filter_area_box.isChecked():
            self.blob_detector_params.minArea = float(self.blob_min_area.text())
            self.blob_detector_params.maxArea = float(self.blob_max_area.text())

        self.blob_detector_params.filterByInertia = self.blob_filter_inertia_box.isChecked()
        if self.blob_filter_inertia_box.isChecked():
            self.blob_detector_params.minInertiaRatio = float(self.blob_min_inertia.text()) + 0.1
            self.blob_detector_params.maxInertiaRatio = float(self.blob_max_inertia.text()) + 0.1

        self.blob_detector_params.filterByColor = self.blob_filter_color_box.isChecked()
        if self.blob_filter_color_box.isChecked():
            self.blob_detector_params.blobColor = int(self.blob_blob_color.text())

        # enable blobber
        detector = cv2.SimpleBlobDetector_create(self.blob_detector_params)
        z = int(self.z_comboBox.currentText())

        # input computing
        keypoints_in = detector.detect(self.input_image[:, :, z].astype(np.uint8))
        keypoints_in = [(round(key.pt[0]), round(key.pt[1])) for key in keypoints_in]
        print(f'[ {len(keypoints_in)} DETECTED FOR INPUT ]')
        input_with_keypoints = cv2.cvtColor(self.input_image[:, :, z].astype(np.uint8), cv2.COLOR_GRAY2RGB)
        for p in keypoints_in:
            input_with_keypoints = cv2.circle(input_with_keypoints, (p[0], p[1]), radius=3, color=(255,0,0), thickness=-1)

        # output computing
        keypoints_out = detector.detect(self.output_image[:, :, z].astype(np.uint8))
        keypoints_out = [(round(key.pt[0]), round(key.pt[1])) for key in keypoints_out]
        print(f'[ {len(keypoints_out)} DETECTED FOR OUTPUT ]')
        output_with_keypoints = cv2.cvtColor(self.output_image[:, :, z].astype(np.uint8), cv2.COLOR_GRAY2RGB)
        for p in keypoints_out:
            output_with_keypoints = cv2.circle(output_with_keypoints, (p[0], p[1]), radius=3, color=(255,0,0), thickness=-1)

        # noise
        input_points = set(keypoints_in)
        output_points = set(keypoints_out)
        noise_points = input_points.intersection(output_points)
        total_point = input_points.union(output_points)
        noise_keys = list(total_point - noise_points)

        print(f'[ {len(noise_keys)} NOISE POINTS DETECTED ]')
        for p in noise_keys:
            output_with_keypoints = cv2.circle(output_with_keypoints, (p[0], p[1]), radius=3, color=(255,255,0), thickness=-1)

        # view
        self.input_slice.setPixmap(self.print_image(input_with_keypoints, is_3d=False))
        self.output_slice.setPixmap(self.print_image(output_with_keypoints, is_3d=False))

    def create_model(self):
        """Create Model routine

        :return:
        """
        print("[ MODEL CREATED ]")
        if self.model_comboBox.currentText() == 'unet':
            self.model = UNET(DEPTH, DEPTH)
        if self.model_comboBox.currentText() == 'vit':
            self.model = SegFormer(
                in_channels=10,
                widths=[64, 128, 256, 512],
                depths=[3, 4, 6, 3],
                all_num_heads=[1, 2, 4, 8],
                patch_size=[7, 3, 3, 3],
                overlap_sizes=[4, 2, 2, 2],
                reduction_ratios=[8, 4, 2, 1],
                mlp_expansions=[4, 4, 4, 4],
                decoder_channels=256,
                scale_factors=[8, 4, 2, 1],
                out_channels=10
            )
        self.model_loaded = True

    def load_experiment(self):
        """Load experiment for model

        :return:
        """
        print("[ EXPERIMENT LOADED ]")
        file = os.path.join(self.experiment_comboBox.currentText(), f'{self.experiment_comboBox.currentText()}.pth.tar')
        checkpoint = torch.load(os.path.join(EXPERIMENTS_PATH, file), map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])

    def on_model_changed(self):
        self.create_model()

    def on_experiment_changed(self):
        if not self.model_loaded:
            self.create_model()
        self.load_experiment()

        if self.output_image is not None and self.input_image is not None:
            self.output_slice.setPixmap(self.print_image(self.output_image.copy()))
            self.input_slice.setPixmap(self.print_image(self.input_image.copy()))

    def on_snr_changed(self):
        if not self.model_loaded: return
        self.load_results()
        self.input_slice.setPixmap(self.print_image(self.input_image.copy()))
        self.output_slice.setPixmap(self.print_image(self.output_image.copy()))

    def on_density_changed(self):
        if not self.model_loaded: return
        self.load_results()
        self.input_slice.setPixmap(self.print_image(self.input_image.copy()))
        self.output_slice.setPixmap(self.print_image(self.output_image.copy()))

    def on_t_changed(self):
        if not self.model_loaded: return
        self.load_results()
        self.input_slice.setPixmap(self.print_image(self.input_image.copy()))
        self.output_slice.setPixmap(self.print_image(self.output_image.copy()))

    def on_slice_changed(self):
        if not self.model_loaded: return
        if self.input_image is None: return
        if self.output_image is None: return
        self.input_slice.setPixmap(self.print_image(self.input_image.copy()))
        self.output_slice.setPixmap(self.print_image(self.output_image.copy()))

    def on_blob_clicked(self):
        if not self.model_loaded: return
        if self.input_image is None: return
        if self.output_image is None: return

        # hande enable-disable
        self.blob_activate = not self.blob_activate
        if self.blob_activate:
            self.blob_button.setStyleSheet("background-color : green")
        else:
            self.blob_button.setStyleSheet("background-color : red")
            self.output_slice.setPixmap(self.print_image(self.output_image))
            self.input_slice.setPixmap(self.print_image(self.input_image))
        self.set_blobber_params()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(955, 661)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.blobber_layout = QtWidgets.QHBoxLayout()
        self.blobber_layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.blobber_layout.setObjectName("blobber_layout")
        self.images_Layout = QtWidgets.QHBoxLayout()
        self.images_Layout.setContentsMargins(15, 15, 15, 15)
        self.images_Layout.setObjectName("images_Layout")
        self.input_slice = QtWidgets.QLabel(self.centralwidget)
        self.input_slice.setText("")
        self.input_slice.setObjectName("input_slice")
        self.images_Layout.addWidget(self.input_slice)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.images_Layout.addItem(spacerItem)
        self.output_slice = QtWidgets.QLabel(self.centralwidget)
        self.output_slice.setText("")
        self.output_slice.setObjectName("output_slice")
        self.images_Layout.addWidget(self.output_slice)
        self.images_Layout.setStretch(0, 1)
        self.images_Layout.setStretch(2, 1)
        self.blobber_layout.addLayout(self.images_Layout)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.blobber_layout.addWidget(self.line)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.blob_min_threshold = QtWidgets.QLineEdit(self.centralwidget)
        self.blob_min_threshold.setObjectName("blob_min_threshold")
        self.gridLayout.addWidget(self.blob_min_threshold, 0, 1, 1, 1)
        self.blob_filter_convexity_lbl = QtWidgets.QLabel(self.centralwidget)
        self.blob_filter_convexity_lbl.setObjectName("blob_filter_convexity_lbl")
        self.gridLayout.addWidget(self.blob_filter_convexity_lbl, 5, 0, 1, 1)
        self.blob_max_inertia = QtWidgets.QLineEdit(self.centralwidget)
        self.blob_max_inertia.setText("")
        self.blob_max_inertia.setObjectName("blob_max_inertia")
        self.gridLayout.addWidget(self.blob_max_inertia, 13, 1, 1, 1)
        self.blob_max_circularity_lbl = QtWidgets.QLabel(self.centralwidget)
        self.blob_max_circularity_lbl.setObjectName("blob_max_circularity_lbl")
        self.gridLayout.addWidget(self.blob_max_circularity_lbl, 4, 0, 1, 1)
        self.blob_min_threshold_lbl = QtWidgets.QLabel(self.centralwidget)
        self.blob_min_threshold_lbl.setObjectName("blob_min_threshold_lbl")
        self.gridLayout.addWidget(self.blob_min_threshold_lbl, 0, 0, 1, 1)
        self.blob_filter_circularity_box = QtWidgets.QCheckBox(self.centralwidget)
        self.blob_filter_circularity_box.setText("")
        self.blob_filter_circularity_box.setObjectName("blob_filter_circularity_box")
        self.gridLayout.addWidget(self.blob_filter_circularity_box, 2, 1, 1, 1)
        self.blob_filter_inertia_lbl = QtWidgets.QLabel(self.centralwidget)
        self.blob_filter_inertia_lbl.setObjectName("blob_filter_inertia_lbl")
        self.gridLayout.addWidget(self.blob_filter_inertia_lbl, 11, 0, 1, 1)
        self.blob_max_circularity = QtWidgets.QLineEdit(self.centralwidget)
        self.blob_max_circularity.setText("")
        self.blob_max_circularity.setObjectName("blob_max_circularity")
        self.gridLayout.addWidget(self.blob_max_circularity, 4, 1, 1, 1)
        self.blob_filter_area_box = QtWidgets.QCheckBox(self.centralwidget)
        self.blob_filter_area_box.setText("")
        self.blob_filter_area_box.setObjectName("blob_filter_area_box")
        self.gridLayout.addWidget(self.blob_filter_area_box, 8, 1, 1, 1)
        self.blob_max_threshold_lbl = QtWidgets.QLabel(self.centralwidget)
        self.blob_max_threshold_lbl.setObjectName("blob_max_threshold_lbl")
        self.gridLayout.addWidget(self.blob_max_threshold_lbl, 1, 0, 1, 1)
        self.blob_min_circularity = QtWidgets.QLineEdit(self.centralwidget)
        self.blob_min_circularity.setText("")
        self.blob_min_circularity.setObjectName("blob_min_circularity")
        self.gridLayout.addWidget(self.blob_min_circularity, 3, 1, 1, 1)
        self.blob_filter_circularity_lbl = QtWidgets.QLabel(self.centralwidget)
        self.blob_filter_circularity_lbl.setObjectName("blob_filter_circularity_lbl")
        self.gridLayout.addWidget(self.blob_filter_circularity_lbl, 2, 0, 1, 1)
        self.blob_min_inertia = QtWidgets.QLineEdit(self.centralwidget)
        self.blob_min_inertia.setText("")
        self.blob_min_inertia.setObjectName("blob_min_inertia")
        self.gridLayout.addWidget(self.blob_min_inertia, 12, 1, 1, 1)
        self.blob_filter_color_box = QtWidgets.QCheckBox(self.centralwidget)
        self.blob_filter_color_box.setText("")
        self.blob_filter_color_box.setObjectName("blob_filter_color_box")
        self.gridLayout.addWidget(self.blob_filter_color_box, 14, 1, 1, 1)
        self.blob_filter_inertia_box = QtWidgets.QCheckBox(self.centralwidget)
        self.blob_filter_inertia_box.setText("")
        self.blob_filter_inertia_box.setObjectName("blob_filter_inertia_box")
        self.gridLayout.addWidget(self.blob_filter_inertia_box, 11, 1, 1, 1)
        self.blob_max_threshold = QtWidgets.QLineEdit(self.centralwidget)
        self.blob_max_threshold.setObjectName("blob_max_threshold")
        self.gridLayout.addWidget(self.blob_max_threshold, 1, 1, 1, 1)
        self.blob_min_convexity_lbl = QtWidgets.QLabel(self.centralwidget)
        self.blob_min_convexity_lbl.setObjectName("blob_min_convexity_lbl")
        self.gridLayout.addWidget(self.blob_min_convexity_lbl, 6, 0, 1, 1)
        self.blob_min_convexity = QtWidgets.QLineEdit(self.centralwidget)
        self.blob_min_convexity.setText("")
        self.blob_min_convexity.setObjectName("blob_min_convexity")
        self.gridLayout.addWidget(self.blob_min_convexity, 6, 1, 1, 1)
        self.blob_max_area_lbl = QtWidgets.QLabel(self.centralwidget)
        self.blob_max_area_lbl.setObjectName("blob_max_area_lbl")
        self.gridLayout.addWidget(self.blob_max_area_lbl, 10, 0, 1, 1)
        self.blob_min_circularity_lbl = QtWidgets.QLabel(self.centralwidget)
        self.blob_min_circularity_lbl.setObjectName("blob_min_circularity_lbl")
        self.gridLayout.addWidget(self.blob_min_circularity_lbl, 3, 0, 1, 1)
        self.blob_filter_convexity_box = QtWidgets.QCheckBox(self.centralwidget)
        self.blob_filter_convexity_box.setText("")
        self.blob_filter_convexity_box.setObjectName("blob_filter_convexity_box")
        self.gridLayout.addWidget(self.blob_filter_convexity_box, 5, 1, 1, 1)
        self.blob_max_convexity_lbl = QtWidgets.QLabel(self.centralwidget)
        self.blob_max_convexity_lbl.setObjectName("blob_max_convexity_lbl")
        self.gridLayout.addWidget(self.blob_max_convexity_lbl, 7, 0, 1, 1)
        self.blob_max_inertia_lbl = QtWidgets.QLabel(self.centralwidget)
        self.blob_max_inertia_lbl.setObjectName("blob_max_inertia_lbl")
        self.gridLayout.addWidget(self.blob_max_inertia_lbl, 13, 0, 1, 1)
        self.blob_max_convexity = QtWidgets.QLineEdit(self.centralwidget)
        self.blob_max_convexity.setText("")
        self.blob_max_convexity.setObjectName("blob_max_convexity")
        self.gridLayout.addWidget(self.blob_max_convexity, 7, 1, 1, 1)
        self.blob_min_area_lbl = QtWidgets.QLabel(self.centralwidget)
        self.blob_min_area_lbl.setObjectName("blob_min_area_lbl")
        self.gridLayout.addWidget(self.blob_min_area_lbl, 9, 0, 1, 1)
        self.blob_min_inertia_lbl = QtWidgets.QLabel(self.centralwidget)
        self.blob_min_inertia_lbl.setObjectName("blob_min_inertia_lbl")
        self.gridLayout.addWidget(self.blob_min_inertia_lbl, 12, 0, 1, 1)
        self.blob_filter_color_lbl = QtWidgets.QLabel(self.centralwidget)
        self.blob_filter_color_lbl.setObjectName("blob_filter_color_lbl")
        self.gridLayout.addWidget(self.blob_filter_color_lbl, 14, 0, 1, 1)
        self.blob_filter_area_lbl = QtWidgets.QLabel(self.centralwidget)
        self.blob_filter_area_lbl.setObjectName("blob_filter_area_lbl")
        self.gridLayout.addWidget(self.blob_filter_area_lbl, 8, 0, 1, 1)
        self.blob_max_area = QtWidgets.QLineEdit(self.centralwidget)
        self.blob_max_area.setText("")
        self.blob_max_area.setObjectName("blob_max_area")
        self.gridLayout.addWidget(self.blob_max_area, 10, 1, 1, 1)
        self.blob_min_area = QtWidgets.QLineEdit(self.centralwidget)
        self.blob_min_area.setText("")
        self.blob_min_area.setObjectName("blob_min_area")
        self.gridLayout.addWidget(self.blob_min_area, 9, 1, 1, 1)
        self.blob_blob_color_lbl = QtWidgets.QLabel(self.centralwidget)
        self.blob_blob_color_lbl.setObjectName("blob_blob_color_lbl")
        self.gridLayout.addWidget(self.blob_blob_color_lbl, 15, 0, 1, 1)
        self.blob_blob_color = QtWidgets.QLineEdit(self.centralwidget)
        self.blob_blob_color.setText("")
        self.blob_blob_color.setObjectName("blob_blob_color")
        self.gridLayout.addWidget(self.blob_blob_color, 15, 1, 1, 1)
        self.blobber_layout.addLayout(self.gridLayout)
        self.blobber_layout.setStretch(0, 1)
        self.verticalLayout_2.addLayout(self.blobber_layout)

        # blob button
        self.blob_button = QtWidgets.QPushButton(self.centralwidget)
        self.blob_button.setObjectName("blob_button")
        self.blob_button.setStyleSheet("background-color : red")
        self.blob_button.clicked.connect(self.on_blob_clicked)

        self.verticalLayout_2.addWidget(self.blob_button)
        self.gridLayout_2.addLayout(self.verticalLayout_2, 4, 0, 1, 1)
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.gridLayout_2.addWidget(self.line_2, 2, 0, 1, 1)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.model_label = QtWidgets.QLabel(self.centralwidget)
        self.model_label.setObjectName("model_label")
        self.horizontalLayout.addWidget(self.model_label)

        # model
        self.model_comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.model_comboBox.setObjectName("model_comboBox")
        self.horizontalLayout.addWidget(self.model_comboBox)
        self.model_comboBox.addItems(["unet", "vit"])
        self.model_comboBox.currentTextChanged.connect(self.on_model_changed)

        self.horizontalLayout.addWidget(self.model_comboBox)
        self.horizontalLayout_7.addLayout(self.horizontalLayout)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.experiment_label = QtWidgets.QLabel(self.centralwidget)
        self.experiment_label.setObjectName("experiment_label")
        self.horizontalLayout_2.addWidget(self.experiment_label)

        # experiment
        self.experiment_comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.experiment_comboBox.setObjectName("experiment_comboBox")
        self.experiment_comboBox.addItems(os.listdir(EXPERIMENTS_PATH))
        self.experiment_comboBox.currentTextChanged.connect(self.on_experiment_changed)

        self.horizontalLayout_2.addWidget(self.experiment_comboBox)
        self.horizontalLayout_7.addLayout(self.horizontalLayout_2)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.snr_label = QtWidgets.QLabel(self.centralwidget)
        self.snr_label.setObjectName("snr_label")
        self.horizontalLayout_3.addWidget(self.snr_label)

        # snr
        self.snr_comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.snr_comboBox.setObjectName("snr_comboBox")
        self.snr_comboBox.addItems(mapSNR.keys())
        self.snr_comboBox.currentTextChanged.connect(self.on_snr_changed)

        self.horizontalLayout_3.addWidget(self.snr_comboBox)
        self.horizontalLayout_7.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem3)
        self.density_label = QtWidgets.QLabel(self.centralwidget)
        self.density_label.setObjectName("density_label")
        self.horizontalLayout_4.addWidget(self.density_label)

        # density
        self.density_comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.density_comboBox.setObjectName("density_comboBox")
        self.density_comboBox.addItems(mapDensity.keys())
        self.density_comboBox.currentTextChanged.connect(self.on_density_changed)

        self.horizontalLayout_4.addWidget(self.density_comboBox)
        self.horizontalLayout_7.addLayout(self.horizontalLayout_4)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.t_label = QtWidgets.QLabel(self.centralwidget)
        self.t_label.setObjectName("t_label")
        self.horizontalLayout_5.addWidget(self.t_label)

        # t
        self.t_comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.t_comboBox.setObjectName("t_comboBox")
        self.t_comboBox.addItems([str(i) for i in range(TIME_INTERVAL)])
        self.t_comboBox.currentTextChanged.connect(self.on_t_changed)

        self.horizontalLayout_5.addWidget(self.t_comboBox)
        self.horizontalLayout_7.addLayout(self.horizontalLayout_5)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem5)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.z_label = QtWidgets.QLabel(self.centralwidget)
        self.z_label.setObjectName("z_label")
        self.horizontalLayout_6.addWidget(self.z_label)

        # z
        self.z_comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.z_comboBox.setObjectName("z_comboBox")
        self.z_comboBox.addItems([str(i) for i in range(DEPTH)])
        self.z_comboBox.currentTextChanged.connect(self.on_slice_changed)

        self.horizontalLayout_6.addWidget(self.z_comboBox)
        self.horizontalLayout_7.addLayout(self.horizontalLayout_6)
        self.gridLayout_2.addLayout(self.horizontalLayout_7, 0, 0, 1, 1)
        spacerItem6 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem6, 3, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        # set default values for blob
        self.blob_min_circularity.setText(str(0.1))
        self.blob_max_circularity.setText(str(1.0))
        self.blob_min_convexity.setText(str(0.1))
        self.blob_max_convexity.setText(str(1.0))
        self.blob_min_area.setText(str(5))
        self.blob_max_area.setText(str(49))
        self.blob_min_inertia.setText(str(0.1))
        self.blob_max_inertia.setText(str(1.0))
        self.blob_blob_color.setText(str(255))

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Blob Detector"))
        self.blob_min_threshold.setText(_translate("MainWindow", "0"))
        self.blob_filter_convexity_lbl.setText(_translate("MainWindow", "filter by convexity:"))
        self.blob_max_circularity_lbl.setText(_translate("MainWindow", "max circularity:"))
        self.blob_min_threshold_lbl.setText(_translate("MainWindow", "Min Threshold:"))
        self.blob_filter_inertia_lbl.setText(_translate("MainWindow", "filter by inertia:"))
        self.blob_max_threshold_lbl.setText(_translate("MainWindow", "Max Threshold:"))
        self.blob_filter_circularity_lbl.setText(_translate("MainWindow", "filter by circularity:"))
        self.blob_max_threshold.setText(_translate("MainWindow", "100"))
        self.blob_min_convexity_lbl.setText(_translate("MainWindow", "min convexity:"))
        self.blob_max_area_lbl.setText(_translate("MainWindow", "max area:"))
        self.blob_min_circularity_lbl.setText(_translate("MainWindow", "min circularity:"))
        self.blob_max_convexity_lbl.setText(_translate("MainWindow", "max convexity:"))
        self.blob_max_inertia_lbl.setText(_translate("MainWindow", "max inertia:"))
        self.blob_min_area_lbl.setText(_translate("MainWindow", "min area:"))
        self.blob_min_inertia_lbl.setText(_translate("MainWindow", "min inertia:"))
        self.blob_filter_color_lbl.setText(_translate("MainWindow", "filter by color:"))
        self.blob_filter_area_lbl.setText(_translate("MainWindow", "filter by area:"))
        self.blob_blob_color_lbl.setText(_translate("MainWindow", "blob color:"))
        self.blob_button.setText(_translate("MainWindow", "Enable and Disable Blob Detector"))
        self.model_label.setText(_translate("MainWindow", "Model:"))
        self.experiment_label.setText(_translate("MainWindow", "Experiment:"))
        self.snr_label.setText(_translate("MainWindow", "SNR:"))
        self.density_label.setText(_translate("MainWindow", "Density:"))
        self.t_label.setText(_translate("MainWindow", "t:"))
        self.z_label.setText(_translate("MainWindow", "z:"))
