"""Parser for json files"""

import os
import json


def json_parser_train(path_file: str):
    """Json parser to take settings for training

    :param path_file:
    :return:
    """
    if not os.path.exists(path_file): return {}, {}, {}
    with open(path_file, 'r') as json_file:
        json_data = json.load(json_file)
        settings = json_data['settings']
        training_dts = json_data['training_dts']
        hyperparameters = json_data['hyperparameters']
    return settings, training_dts, hyperparameters


def json_parser_inference(path_file: str):
    """Json parser to take dataset to use

    :param path_file:
    :return:
    """
    if not os.path.exists(path_file): return "", "", []
    with open(path_file, 'r') as json_file:
        json_data = json.load(json_file)
        model = json_data['model']
        params = json_data['params']
        dataset = json_data['dataset']
    return model, params, dataset
