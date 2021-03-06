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
import numpy as np

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
    parser.set_defaults(data_padding=True)

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
    parser.add_option('--data_padding', dest='data_padding')

    parser.add_option('--use_pit', dest='use_pit', action='store_true')
    parser.add_option('--no_pit', dest='use_pit', action='store_false')
    parser.set_defaults(use_pit=True)

    parser.add_option('--use_pad', dest='zero_pad', action='store_true')
    parser.add_option('--no_pad', dest='zero_pad', action='store_false')
    parser.set_defaults(zero_pad=True)

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
    train_set_generator = dataset.get_random_batch_generator('train', pad=cla.zero_pad)
    valid_set_generator = dataset.get_random_batch_generator('valid', pad=cla.zero_pad)

    # num_train_samples = 10 #dataset.get_num_batch_in_dataset('train')
    # num_valid_samples = 10 #dataset.get_num_batch_in_dataset('valid')

    model.fit_model(train_set_generator, num_train_samples, valid_set_generator, num_valid_samples, \
                    config['training']['num_epochs'])
    # model.fit_model(train_set_generator, 10, valid_set_generator, 2, config['training']['num_epochs'])


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


def test(config, cla):

    if cla.batch_size is not None:
        batch_size = int(cla.batch_size)
    else:
        batch_size = config['training']['batch_size']

    if cla.target_field_length is not None:
        cla.target_field_length = int(cla.target_field_length)

    model = models.DenoisingWavenet(config, target_field_length=cla.target_field_length,
                                    load_checkpoint=cla.load_checkpoint, print_model_summary=cla.print_model_summary)

    samples_folder_path = os.path.join(config['training']['path'], 'samples')
    output_folder_path = get_valid_output_folder_path(samples_folder_path)

    if not cla.noisy_input_path.endswith('/'):
        cla.noisy_input_path += '/'
    filenames = [filename for filename in os.listdir(cla.noisy_input_path) if filename.endswith('.wav')]

    with open('spk_info.json') as f:
        spk_info = json.load(f)
    
    sdr = []
    n_output = config['training']['n_output'] if 'n_output' in config['training'] else 2
    n_speaker = config['training']['n_speaker'] if 'n_speaker' in config['training'] else 2
    gender_stat = {'ch'+str(i+1):{'M':0,'F':0} for i in range(n_output)}
    # gender_stat = {'ch1':{'M':0,'F':0}, 'ch2':{'M':0,'F':0}}

    for filename in filenames:
        noisy_input = util.load_wav(cla.noisy_input_path + filename, config['dataset']['sample_rate'])
        if cla.clean_input_path is not None:
            if not cla.clean_input_path.endswith('/'):
                cla.clean_input_path += '/'
            clean_input_1 = util.load_wav(cla.clean_input_path + 's1/' + filename, config['dataset']['sample_rate'])
            clean_input_2 = util.load_wav(cla.clean_input_path + 's2/' + filename, config['dataset']['sample_rate'])
        input = {'noisy': noisy_input, 'clean_1': clean_input_1, 'clean_2':clean_input_2}

        output_filename_prefix = filename[0:-4] + '_'
        spk1 = output_filename_prefix.split('_')[0][:3]
        spk2 = output_filename_prefix.split('_')[2][:3]
        spk_name = [spk1, spk2]
        spk_gender = [spk_info[spk1], spk_info[spk2]]

        # print("Denoising: " + filename).
        condition_input = None
        print(filename)
        _sdr, ch_gender, pit_idx = denoise.denoise_sample(model, input, condition_input, batch_size, output_filename_prefix,
                                      config['dataset']['sample_rate'], n_speaker, n_output, output_folder_path, 
                                      spk_gender=spk_gender,
                                      use_pit=cla.use_pit, pad=cla.zero_pad)
        # print('sdr = %f, %f' %(_sdr[0],_sdr[1]))
        if spk_gender[0] == 'F' and spk_gender[1] == 'M':
            for i in range(1, -1, -1):
                print('{} {}: sdr={}, idx={}'.format(spk_gender[i], spk_name[i], _sdr[i], pit_idx[i]))
        else:
            for i in range(2):
                print('{} {}: sdr={}, idx={}'.format(spk_gender[i], spk_name[i], _sdr[i], pit_idx[i]))
        # print(ch_gender)
        # for ch, stat in ch_gender.items():
            # for gen, num in stat.items():
                # gender_stat[ch][gen] += num
        sdr.append(_sdr)
    sdr = np.array(sdr)
    print('Testing SDR:', np.mean(sdr))
    print(gender_stat)

def inference(config, cla):

    if cla.batch_size is not None:
        batch_size = int(cla.batch_size)
    else:
        batch_size = config['training']['batch_size']

    if cla.target_field_length is not None:
        cla.target_field_length = int(cla.target_field_length)

    if not bool(cla.one_shot):
        model = models.DenoisingWavenet(config, target_field_length=cla.target_field_length,
                                        load_checkpoint=cla.load_checkpoint, print_model_summary=cla.print_model_summary)
        print('Performing inference..')
    else:
        print('Performing one-shot inference..')

    samples_folder_path = os.path.join(config['training']['path'], 'samples')
    output_folder_path = get_valid_output_folder_path(samples_folder_path)

    #If input_path is a single wav file, then set filenames to single element with wav filename
    if cla.noisy_input_path.endswith('.wav'):
        filenames = [cla.noisy_input_path.rsplit('/', 1)[-1]]
        cla.noisy_input_path = cla.noisy_input_path.rsplit('/', 1)[0] + '/'
        if cla.clean_input_path is not None:
            cla.clean_input_path = cla.clean_input_path.rsplit('/', 1)[0] + '/'
    else:
        if not cla.noisy_input_path.endswith('/'):
            cla.noisy_input_path += '/'
        filenames = [filename for filename in os.listdir(cla.noisy_input_path) if filename.endswith('.wav')]

    clean_input = None
    for filename in filenames:
        noisy_input = util.load_wav(cla.noisy_input_path + filename, config['dataset']['sample_rate'])
        if cla.clean_input_path is not None:
            if not cla.clean_input_path.endswith('/'):
                cla.clean_input_path += '/'
            clean_input_1 = util.load_wav(cla.clean_input_path + 's1/' + filename, config['dataset']['sample_rate'])
            clean_input_2 = util.load_wav(cla.clean_input_path + 's2/' + filename, config['dataset']['sample_rate'])
        input = {'noisy': noisy_input, 'clean_1': clean_input_1, 'clean_2':clean_input_2}

        output_filename_prefix = filename[0:-4] + '_'

        if bool(cla.one_shot):
            if len(input['noisy']) % 2 == 0:  # If input length is even, remove one sample
                input['noisy'] = input['noisy'][:-1]
                if input['clean'] is not None:
                    input['clean'] = input['clean'][:-1]
            model = models.DenoisingWavenet(config, load_checkpoint=cla.load_checkpoint, input_length=len(input['noisy']), \
                                            print_model_summary=cla.print_model_summary)
        condition_input = None
        print("Denoising: " + filename)
        denoise.denoise_sample(model, input, condition_input, batch_size, output_filename_prefix,
                                            config['dataset']['sample_rate'], output_folder_path,
                                            save_wav=True)


def main():

    set_system_settings()
    cla = get_command_line_arguments()
    config = load_config(cla.config)
    config['training']['use_condition'] = cla.use_condition

    if cla.batch_size != None:
        config['training']['batch_size'] = int(cla.batch_size)
    if cla.target_field_length != None:
        config['model']['target_field_length'] = int(cla.target_field_length)
    print('Using Batch Size:', config['training']['batch_size'])
    print('Target Field Length:', config['model']['target_field_length'])
    if cla.mode == 'training':
        training(config, cla)
    elif cla.mode == 'inference':
        inference(config, cla)
    elif cla.mode == 'test':
        test(config, cla)


if __name__ == "__main__":
    main()
