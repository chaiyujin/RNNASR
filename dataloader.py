import os
import pickle
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
    phoneme_list = []
    phoneme_map = {}
    with open(path) as file:
        idx = 0
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
        with open(file_prefix + '.PHN') as file:
            for line in file:
                phn = line.strip().split(' ')[2]
                d['phn'].append(map[phn])
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


if __name__ == '__main__':
    # find_all_phonemes('../../dataset/TIMIT/')
    # load_timit('../../dataset/TIMIT/TRAIN', './train.pkl')
    # load_timit('../../dataset/TIMIT/TEST', './test.pkl')

    all_data = load_data()
    print('Train set: ' + str(len(all_data['train_set'])))
    print('Test  set: ' + str(len(all_data['test_set'])))
