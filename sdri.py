# A Wavenet For Speech Denoising - Dario Rethage - 19.05.2017
# SDR_i

from tqdm import tqdm
import soundfile as sf
import os
import mir_eval
import numpy as np

noisy_path = './wsj0-mix/2speakers/wav8k/max/tt/mix/'
single_path = './wsj0-mix/2speakers/wav8k/max/tt/'

filenames = [filename for filename in os.listdir(noisy_path) if filename.endswith('.wav')]

def sdr_cal(y1, y2):
    return 10 * np.log10(np.dot(y1,y2) ** 2 / (np.dot(y1, y1) * np.dot(y2, y2) - np.dot(y1,y2) ** 2))

sdr = []
mysdr = []

for filename in tqdm(filenames):
    noisy_wav, smp_rate = sf.read(noisy_path + filename)
    clean_1, smp_rate = sf.read(single_path + 's1/' + filename)
    clean_2, smp_rate = sf.read(single_path + 's2/' + filename)
    _sdr1, _sir, _sar, _popt = mir_eval.separation.bss_eval_sources(clean_1, noisy_wav)
    _sdr2, _sir, _sar, _popt = mir_eval.separation.bss_eval_sources(clean_2, noisy_wav)
    sdr.append([_sdr1, _sdr2])
    mysdr1 = sdr_cal(clean_1, noisy_wav)
    mysdr2 = sdr_cal(clean_2, noisy_wav)
    mysdr.append([mysdr1, mysdr2])
    # print(_sdr1, _sdr2)
    # print(mysdr1, mysdr2)

sdr = np.array(sdr)
mysdr = np.array(mysdr)
print('sdr mean: %f, mysdr mean: %f' % (np.mean(sdr), np.mean(mysdr)))
print('sdr max : %f, mysdr max : %f' % (np.max(sdr), np.max(mysdr)))
print('sdr min : %f, mysdr min : %f' % (np.min(sdr), np.min(mysdr)))
