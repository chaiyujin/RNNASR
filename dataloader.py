import os
import pickle
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc

global_map = {}


def save_phonemes(path='./phonemes'):
    # save the map of id and phoneme
    phoneme_list = []
    for k in global_map:
        phoneme_list.append(k)
    phoneme_list.sort()
    # save in file
    with open('phonemes', 'w') as file:
        for i in range(len(phoneme_list)):
            file.write(str(phoneme_list[i]) + ' ' + str(i) + '\n')


def get_phonemes_list_and_map(path='./phonemes'):
    phoneme_list = ['_']
    phoneme_map = {'_': 0}
    with open(path) as file:
        idx = 1
        for line in file:
            phoneme_list.append(line.strip().split(' ')[0])
            phoneme_map[line.strip().split(' ')[0]] = idx
            idx += 1
    return phoneme_list, phoneme_map


def find_data(root_dir, result_list):
    for parent, dirs, files in os.walk(root_dir):
        for file in files:
            name, ext = os.path.splitext(os.path.join(parent, file))
            if ext == '.WAV':
                result_list.append(name)


def load_timit(data_path, save_path):
    all_data = []
    # 1. get the file name from path
    print('Find data from ', data_path)
    file_list = []
    find_data(data_path, file_list)
    # 2. generate mfcc from audio file and phoneme
    _, map = get_phonemes_list_and_map()
    print('Pre-process data..')
    for file_prefix in file_list:
        print(file_prefix)
        d = {}
        wav_file = file_prefix + '.WAV'
        fs, audio = wav.read(wav_file)
        d['mel'] = mfcc(audio, samplerate=fs)
        d['phn'] = []
        d['phn_nb'] = []
        with open(file_prefix + '.PHN') as file:
            d['phn'].append(0)  # head space
            for line in file:
                phn = line.strip().split(' ')[2]
                assert(map[phn] != 0)
                d['phn'].append(map[phn])
                d['phn'].append(0)  # space after every character
                d['phn_nb'].append(map[phn] - 1)  # no blank at 0
        assert(len(d['phn']) > 0)
        all_data.append(d)

    # print(all_data)
    with open(save_path, 'wb') as pkl_file:
        pickle.dump(all_data, pkl_file)

    # # test load
    # with open(save_path, 'rb') as pkl_file:
    #     data = pickle.load(pkl_file)
    #     print(data)


def find_all_phonemes(root_dir):
    global global_map
    global_map = {}
    result_list = []
    for parent, dirs, files in os.walk(root_dir):
        for file in files:
            name, ext = os.path.splitext(os.path.join(parent, file))
            if ext == '.PHN':
                result_list.append(name)

    for file_p in result_list:
        with open(file_p + '.PHN') as phn_file:
            for line in phn_file:
                phn = line.strip().split(' ')[2]
                if phn in global_map:
                    global_map[phn] += 1
                else:
                    global_map[phn] = 1

    save_phonemes()


def load_data(train_pkl='./train.pkl', test_pkl='./test.pkl'):
    if not os.path.exists(train_pkl):
        load_timit('../../dataset/TIMIT/TRAIN', './train.pkl')
    if not os.path.exists(test_pkl):
        load_timit('../../dataset/TIMIT/TEST', './test.pkl')

    data = {}
    with open(train_pkl, 'rb') as pkl_file:
        data['train_set'] = pickle.load(pkl_file)

    with open(test_pkl, 'rb') as pkl_file:
        data['test_set'] = pickle.load(pkl_file)

    return data


def process_input(data_set):
    inputs = []
    seq_lens = []
    for data in data_set:
        inputs.append(np.asarray(data['mel'], dtype=np.float32))
        seq_lens.append(len(data['mel']))

    return \
        np.asarray(inputs), \
        np.asarray(seq_lens, dtype=np.int32)


def process_target(data_set):
    indices = []
    values = []
    for n, data in enumerate(data_set):
        seq = data['phn_nb']
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.int32)
    shape = np.asarray([len(data_set), indices.max(0)[1] + 1], dtype=np.int64)

    return np.asarray([indices, values, shape])


def process_data(data_set):
    inputs, seq_lens = process_input(data_set)
    targets = process_target(data_set)

    # print('process data.')
    # print('inputs..........')
    # print(inputs)
    # print('seq_lens.......')
    # print(seq_lens)
    # print('targets')
    # print(targets)

    return inputs, seq_lens, targets


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
    # find_all_phonemes('../../dataset/TIMIT/')
    load_timit('../../dataset/TIMIT/TRAIN', './train.pkl')
    load_timit('../../dataset/TIMIT/TEST', './test.pkl')

    all_data = load_data()
    print('Train set: ' + str(len(all_data['train_set'])))
    print('Test  set: ' + str(len(all_data['test_set'])))

    print(all_data['train_set'][0]['phn'])
    print(all_data['train_set'][0]['phn_nb'])
