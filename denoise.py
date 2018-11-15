# A Wavenet For Speech Denoising - Dario Rethage - 19.05.2017
# Denoise.py

from __future__ import division
import os
import util
import tqdm
import numpy as np
import mir_eval

def denoise_sample(model, input, condition_input, batch_size, output_filename_prefix, sample_rate, 
                    output_path, save_wav=False, spk_gender=None, use_pit=False, pad=False):
    pad = True 
    if pad:
        noisy_pad = np.zeros((model.half_receptive_field_length*2 + len(input['noisy'])))
        noisy_pad[model.half_receptive_field_length:model.half_receptive_field_length+len(input['noisy'])] = input['noisy']
        input['noisy'] = noisy_pad
    
    if len(input['noisy']) < model.receptive_field_length:
        raise ValueError('Input is not long enough to be used with this model.')

    num_output_samples = input['noisy'].shape[0] - (model.receptive_field_length - 1)
    num_fragments = int(np.ceil(num_output_samples / model.target_field_length))
    num_batches = int(np.ceil(num_fragments / batch_size))

    ch_gender = {'ch1':{'M':0,'F':0}, 'ch2':{'M':0,'F':0}}

    output_1 = []
    output_2 = []
    num_pad_values = 0
    fragment_i = 0
    for batch_i in tqdm.tqdm(range(0, num_batches)):

        if batch_i == num_batches-1: #If its the last batch'
            batch_size = num_fragments - batch_i*batch_size

        # condition_batch = np.array([condition_input, ] * batch_size, dtype='uint8')
        input_batch = np.zeros((batch_size, model.input_length))

        #Assemble batch
        for batch_fragment_i in range(0, batch_size):

            if fragment_i + model.target_field_length > num_output_samples:
                remainder = input['noisy'][fragment_i:]
                current_fragment = np.zeros((model.input_length,))
                current_fragment[:remainder.shape[0]] = remainder
                num_pad_values = model.input_length - remainder.shape[0]
            else:
                current_fragment = input['noisy'][fragment_i:fragment_i + model.input_length]

            input_batch[batch_fragment_i, :] = current_fragment
            fragment_i += model.target_field_length

        # output_1_fragments = model.denoise_batch({'data_input': input_batch, 'condition_input': condition_batch})
        input_batch = np.concatenate([np.expand_dims(input_batch, 0), np.zeros_like(np.expand_dims(input_batch, 0))])
        input_batch = np.transpose(input_batch, (1,0,2))

        output_fragments = model.denoise_batch({'data_input': input_batch})
        
        output_1_fragment = output_fragments[:, 0]
        output_2_fragment = output_fragments[:, 1]
        
        output_1_fragment = output_1_fragment[:, model.target_padding: model.target_padding + model.target_field_length]
        output_1_fragment = output_1_fragment.flatten().tolist()

        output_2_fragment = output_2_fragment[:, model.target_padding: model.target_padding + model.target_field_length]
        output_2_fragment = output_2_fragment.flatten().tolist()
        
        output_1 = output_1 + output_1_fragment
        output_2 = output_2 + output_2_fragment

    output_1 = np.array(output_1)
    output_2 = np.array(output_2)

    if num_pad_values != 0:
        output_1 = output_1[:-num_pad_values]
        output_2 = output_2[:-num_pad_values]

    valid_noisy_signal = input['noisy'][model.half_receptive_field_length:model.half_receptive_field_length + len(output_1)]
    valid_clean_signal_1 = input['clean_1'][
                     model.half_receptive_field_length:model.half_receptive_field_length + len(output_1)] if not pad else input['clean_1']
    valid_clean_signal_2 = input['clean_2'][
                     model.half_receptive_field_length:model.half_receptive_field_length + len(output_1)] if not pad else input['clean_2']

    if use_pit  == True:
        pit_output_1 = []
        pit_output_2 = []
        for f in range(num_fragments):
            c1 = valid_clean_signal_1[f*model.target_field_length:(f+1)*model.target_field_length]
            c2 = valid_clean_signal_2[f*model.target_field_length:(f+1)*model.target_field_length]
            o1 = output_1[f*model.target_field_length:(f+1)*model.target_field_length]
            o2 = output_2[f*model.target_field_length:(f+1)*model.target_field_length]
            perm = np.argmin([np.sum(np.abs(c1-o1)+np.abs(c2-o2)), np.sum(np.abs(c1-o2)+np.abs(c2-o1))])
            if perm == 0:
                ch_gender['ch1'][spk_gender[0]] += 1
                ch_gender['ch2'][spk_gender[1]] += 1
                pit_output_1 += o1.tolist()
                pit_output_2 += o2.tolist()
            else:
                ch_gender['ch1'][spk_gender[1]] += 1
                ch_gender['ch2'][spk_gender[0]] += 1
                pit_output_1 += o2.tolist()
                pit_output_2 += o1.tolist()
        
        clean_wav = np.array([valid_clean_signal_1, valid_clean_signal_2])
        noisy_wav = np.array([pit_output_1, pit_output_2])
        
        _sdr1, _sir, _sar, _popt = mir_eval.separation.bss_eval_sources(np.array(clean_wav[0]), np.array(noisy_wav[0]))
        _sdr2, _sir, _sar, _popt = mir_eval.separation.bss_eval_sources(np.array(clean_wav[1]), np.array(noisy_wav[1]))
   
        return np.array([_sdr1, _sdr2]), ch_gender
    
    else:
        clean_wav = np.array([valid_clean_signal_1, valid_clean_signal_2])
        noisy_wav = np.array([output_1, output_2])
        _sdr1, _sir, _sar, _popt = mir_eval.separation.bss_eval_sources(np.array(clean_wav[0]), np.array(noisy_wav[0]))
        _sdr2, _sir, _sar, _popt = mir_eval.separation.bss_eval_sources(np.array(clean_wav[1]), np.array(noisy_wav[1]))
        _sdr3, _sir, _sar, _popt = mir_eval.separation.bss_eval_sources(np.array(clean_wav[0]), np.array(noisy_wav[1]))
        _sdr4, _sir, _sar, _popt = mir_eval.separation.bss_eval_sources(np.array(clean_wav[1]), np.array(noisy_wav[0]))
        if _sdr1 + _sdr2 > _sdr3 + _sdr4:
            ch_gender['ch1'][spk_gender[0]] += 1
            ch_gender['ch2'][spk_gender[1]] += 1
            return np.array([_sdr1, _sdr2]), ch_gender
        else:
            ch_gender['ch1'][spk_gender[1]] += 1
            ch_gender['ch2'][spk_gender[0]] += 1
            return np.array([_sdr3, _sdr4]), ch_gender

    if save_wav:
        output_original_filename = output_filename_prefix + 'orig.wav'
        output_s1_filename = output_filename_prefix + 's1.wav'
        output_s2_filename = output_filename_prefix + 's2.wav'

        output_original_filepath = os.path.join(output_path, output_original_filename)
        output_s1_filepath = os.path.join(output_path, output_s1_filename)
        output_s2_filepath = os.path.join(output_path, output_s2_filename)
        print(output_denoised_filepath)

        util.write_wav(valid_noisy_signal, output_original_filepath, sample_rate)
        util.write_wav(output_1, output_s1_filepath, sample_rate)
        util.write_wav(output_2, output_s2_filepath, sample_rate)

    #     noise_in_output_1 = output_1 - valid_clean_signal

    #     rms_clean = util.rms(valid_clean_signal)
    #     rms_noise_out = util.rms(noise_in_output_1)
    #     rms_noise_in = util.rms(input['noise'])

    #     new_snr_db = int(np.round(util.snr_db(rms_clean, rms_noise_out)))
    #     initial_snr_db = int(np.round(util.snr_db(rms_clean, rms_noise_in)))

    # else:
    #     output_denoised_filename = output_filename_prefix + 'denoised.wav'
    #     output_noisy_filename = output_filename_prefix + 'noisy.wav'

