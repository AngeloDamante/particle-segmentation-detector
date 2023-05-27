# """Functions to create segmaps from gth"""
#
# import argparse
# from preprocessing.segmentation import Segmenter
# from utils.Types import mapSNR, mapDensity
#
# TIME_INTERVAL = 100
#
# def main():
#     parser = argparse.ArgumentParser()
#
#     # to create segmentation map
#     parser.add_argument("-SNR", "--snr", type=str, default='snr_7', help="chose signal to noise ratio", choices=mapSNR.keys())
#     parser.add_argument("-D", "--density", type=str, default='density_low', help="chose density", choices=mapDensity.keys())
#     parser.add_argument("-T", "--time_interval", type=int, default=TIME_INTERVAL, help="number of images")
#     parser.add_argument("-IMG", "--save_img", type=bool, default=False, help="save image")
#
#     # to set Segmenter
#     parser.add_argument("-S", "--sigma", type=float, default=1.0, help="select a sigma in (0.1, 2.0)")
#     parser.add_argument("-K", "--kernel", type=int, default=3, help="kernel dimension")
#     parser.add_argument("-V", "--value", type=int, default=255, help="value to put on particles coord")
#
#     # parsing
#     args = parser.parse_args()
#     snr = mapSNR[args.snr]
#     density = mapDensity[args.density]
#     time_interval = args.time_interval
#     save_img = args.save_img
#     sigma = args.sigma
#     kernel = args.kernel
#     value = args.value
#
#     # create and use segmenter
#     o_seg = Segmenter(sigma, kernel, value)
#     o_seg.create_dataset(snr, density, time_interval, save_img)
#
#
# if __name__ == '__main__':
#     main()
