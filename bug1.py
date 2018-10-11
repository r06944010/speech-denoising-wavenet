# A Wavenet For Speech Denoising - Dario Rethage - 19.05.2017
# Main.py

import sys
import logging
import optparse
import json
import os
import models
import datasets
import util
import denoise
from tqdm import tqdm

def set_system_settings():
    sys.setrecursionlimit(50000)
    logging.getLogger().setLevel(logging.INFO)


def get_command_line_arguments():
    parser = optparse.OptionParser()
    parser.set_defaults(config='config.json')
    parser.set_defaults(mode='training')
    parser.set_defaults(load_checkpoint=None)
    parser.set_defaults(condition_value=0)
    parser.set_defaults(batch_size=None)
    parser.set_defaults(one_shot=False)
    parser.set_defaults(clean_input_path=None)
    parser.set_defaults(noisy_input_path=None)
    parser.set_defaults(print_model_summary=False)
    parser.set_defaults(target_field_length=None)
    parser.set_defaults(use_condition=False)

    parser.add_option('--mode', dest='mode')
    parser.add_option('--print_model_summary', dest='print_model_summary')
    parser.add_option('--config', dest='config')
    parser.add_option('--load_checkpoint', dest='load_checkpoint')
    parser.add_option('--condition_value', dest='condition_value')
    parser.add_option('--batch_size', dest='batch_size')
    parser.add_option('--one_shot', dest='one_shot')
    parser.add_option('--noisy_input_path', dest='noisy_input_path')
    parser.add_option('--clean_input_path', dest='clean_input_path')
    parser.add_option('--target_field_length', dest='target_field_length')
    parser.add_option('--use_condition', dest='use_condition')

    (options, args) = parser.parse_args()

    return options


def load_config(config_filepath):
    try:
        config_file = open(config_filepath, 'r')
    except IOError:
        logging.error('No readable config file at path: ' + config_filepath)
        exit()
    else:
        with config_file:
            return json.load(config_file)


def get_dataset(config, model):

    if config['dataset']['type'] == 'vctk+demand':
        return datasets.VCTKAndDEMANDDataset(config, model).load_dataset()
    elif config['dataset']['type'] == 'nsdtsea':
        return datasets.NSDTSEADataset(config, model).load_dataset()
    elif config['dataset']['type'] == 'wsj0-mix':
        return datasets.WSJ0(config, model).load_dataset()


def training(config, cla):

    # Instantiate Model
    model = models.DenoisingWavenet(config, load_checkpoint=cla.load_checkpoint, print_model_summary=cla.print_model_summary)
    dataset = get_dataset(config, model)

    num_train_samples = config['training']['num_train_samples'] // config['training']['batch_size']
    num_valid_samples = config['training']['num_valid_samples'] // config['training']['batch_size'] 
    train_set_generator = dataset.get_random_batch_generator('train')
    valid_set_generator = dataset.get_random_batch_generator('valid')


    for tr in tqdm(range(num_train_samples)):
        train_batch = train_set_generator.__next__()
    for cv in tqdm(range(num_valid_samples)):
        valid_batch = valid_set_generator.__next__()
    print('end of processing')


def get_valid_output_folder_path(outputs_folder_path):
    j = 1
    while True:
        output_folder_name = 'samples_%d' % j
        output_folder_path = os.path.join(outputs_folder_path, output_folder_name)
        if not os.path.isdir(output_folder_path):
            os.mkdir(output_folder_path)
            break
        j += 1
    return output_folder_path

def main():

    set_system_settings()
    cla = get_command_line_arguments()
    config = load_config(cla.config)
    config['training']['use_condition'] = cla.use_condition

    if cla.mode == 'training':
        training(config, cla)
    elif cla.mode == 'inference':
        inference(config, cla)


if __name__ == "__main__":
    main()
