import os
import pickle
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc, delta, logfbank
from utils.phonemes import get_phoneme_id


def find_data(root_dir, result_list):
    for parent, dirs, files in os.walk(root_dir):
        for file in files:
            name, ext = os.path.splitext(os.path.join(parent, file))
            if ext == '.WAV':
                result_list.append(name)


def load_timit(data_path, save_path):
    all_data = {
        'sources': [],
        'targets': [],
        'seq_len': []
    }
    # 1. get the file name from path
    print('Find data from ', data_path)
    file_list = []
    find_data(data_path, file_list)
    # 2. generate mfcc from audio file and phoneme
    print('Pre-process data..')
    for file_prefix in file_list:
        print(file_prefix)
        d = {}
        wav_file = file_prefix + '.WAV'
        fs, audio = wav.read(wav_file)
        mfcc_feat = mfcc(audio, samplerate=fs)
        # d_mfcc_feat = delta(mfcc_feat, 2)
        fbank_feat = logfbank(audio, samplerate=fs)
        d['mel'] = []
        for mel, fbank in zip(mfcc_feat, fbank_feat):
            tmp = []
            for a in mel:
                tmp.append(a)
            for a in fbank:
                tmp.append(a)
            d['mel'].append(np.asarray(tmp, dtype=np.float32))
            # print(len(d['mel'][-1]))
        d['phn'] = []
        print(len(d['mel'][-1]))
        with open(file_prefix + '.PHN') as file:
            for line in file:
                phn = line.strip().split(' ')[2]
                id = get_phoneme_id(phn)
                if id is None:
                    continue
                assert(id is not None)
                if len(d['phn']) == 0 or id != d['phn'][-1]:
                    d['phn'].append(id)
        assert(len(d['phn']) > 0)
        all_data['sources'].append(np.asarray(d['mel'], dtype=np.float32))
        all_data['targets'].append(np.asarray(d['phn'], dtype=np.int32))
        all_data['seq_len'].append(len(d['mel']))

    for k in all_data:
        all_data[k] = np.asarray(all_data[k])

    print(type(all_data['sources']))
    print(type(all_data['targets']))

    # print(all_data)
    with open(save_path, 'wb') as pkl_file:
        pickle.dump(all_data, pkl_file)


def load_data(
      train_data_path='../../dataset/TIMIT/TRAIN', train_pkl='data/train.pkl',
      test_data_path='../../dataset/TIMIT/TEST', test_pkl='data/test.pkl'):
    if not os.path.exists(train_pkl):
        load_timit(train_data_path, train_pkl)
    if not os.path.exists(test_pkl):
        load_timit(test_data_path, test_pkl)

    data = {}
    with open(train_pkl, 'rb') as pkl_file:
        data['train_set'] = pickle.load(pkl_file)

    with open(test_pkl, 'rb') as pkl_file:
        data['test_set'] = pickle.load(pkl_file)

    return data


def load_single(file_prefix='../../dataset/TIMIT/TEST/DR1/MDAB0/SX49'):
    mel_seq = []
    fs, audio = wav.read(file_prefix + '.WAV')
    mfcc_feat = mfcc(audio, samplerate=fs)
    # d_mfcc_feat = delta(mfcc_feat, 2)
    fbank_feat = logfbank(audio, samplerate=fs)
    for mel, fbank in zip(mfcc_feat, fbank_feat):
        tmp = []
        for a in mel:
            tmp.append(a)
        for a in fbank:
            tmp.append(a)
        mel_seq.append(np.asarray(tmp, dtype=np.float32))

    the_data = {
        'prefix': file_prefix,
        'sources': np.asarray([np.asarray(mel_seq)]),
        'seq_len': np.asarray([len(mel_seq)])
    }

    return the_data


def mean_std(inputs):
    tmp = []
    for src in inputs:
        for ele in src:
            tmp.append(ele)
    tmp = np.asarray(tmp).flatten()
    return np.mean(tmp), np.std(tmp)


def process_target(data_set):
    indices = []
    values = []
    for n, seq in enumerate(data_set):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.int32)
    shape = np.asarray([len(data_set), indices.max(0)[1] + 1], dtype=np.int64)

    return np.asarray([indices, values, shape])


def pad_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    '''Pads each sequence to the same length: the length of the longest
    sequence.
        If maxlen is provided, any sequence longer than maxlen is truncated to
        maxlen. Truncation happens off either the beginning or the end
        (default) of the sequence. Supports post-padding (default) and
        pre-padding.

        Args:
            sequences: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger
            than maxlen either in the beginning or in the end of the sequence
            value: float, value to pad the sequences to the desired value.
        Returns
            x: numpy array with dimensions (number_of_sequences, maxlen)
            lengths: numpy array with the original sequence lengths
    '''
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, lengths


if __name__ == '__main__':
    load_timit('../../dataset/TIMIT/TRAIN', './data/train.pkl')
    load_timit('../../dataset/TIMIT/TEST', '././data/test.pkl')
    all_data = load_data()
    print(mean_std(all_data['train_set']['sources']))
