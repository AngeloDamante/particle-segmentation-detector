"""Blob Detector Program"""

import sys
from PyQt5 import QtWidgets
from form_blob_detector import Ui_MainWindow


def main():
    # Create the application object
    app = QtWidgets.QApplication(sys.argv)
    my_app = Ui_MainWindow()
    window = QtWidgets.QMainWindow()
    my_app.setupUi(window)

    # run
    window.show()

    # stop
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
