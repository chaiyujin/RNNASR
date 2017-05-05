import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
try:
    import phonemes
except:
    import utils.phonemes as phonemes


def draw_wav(fig, path):
    spf = wave.open(path + '.WAV', 'r')

    # Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    fs = spf.getframerate()

    # If Stereo
    if spf.getnchannels() == 2:
        print('Just mono files')
        sys.exit(0)

    Time = np.linspace(0, len(signal)/fs, num=len(signal))

    # draw wav signal
    wav_plt = fig.add_subplot(313)
    wav_plt.title.set_text('Signal Wave')
    wav_plt.plot(Time, signal)

    # get time-aligned phonemes
    times = []
    phone = []
    with open(path + '.PHN', 'r') as file:
        dura = []
        phns = []
        for line in file:
            splited = line.strip().split(' ')
            start = int(splited[0])
            end = int(splited[1])
            dura.append(end - start)
            id = phonemes.get_phoneme_id(splited[2])
            if id is None:
                id = phns[-1]
            phns.append(id)

        cur = 0
        need = 160
        delta = 0.01
        for i in range(len(dura)):
            while dura[i] >= need:
                times.append(cur)
                phone.append(phns[i])
                cur += delta
                dura[i] -= need
                need = 160
            need -= dura[i]

    phn_plt = fig.add_subplot(312)
    phn_plt.set_yticks(np.arange(len(phonemes.g_phns_list)))
    phn_plt.set_yticklabels(phonemes.g_phns_list, minor=False)
    phn_plt.title.set_text('Phoneme Heatmap')
    phn_plt.scatter(times, phone)


def draw_phoneme(fig, prob):
    idx = [i for i in range(len(prob))]
    probs = [[] for _ in range(40)]

    for i in range(len(prob)):
        cur = prob[i][0]
        for n, val in enumerate(cur):
            probs[n].append(val)

    prob_plt = fig.add_subplot(311)
    for i in range(40):
        phn = '<blank>'
        if i < 39:
            phn = phonemes.g_phns_list[i]
        c = '#' + format(i, '06b')
        c = c.replace('1', 'F')
        if i < 39:
            prob_plt.plot(idx, probs[i], color=c, label=phn)
        else:
            prob_plt.plot(idx, probs[i], 'k-.', label=phn)
    plt.legend(prop={'size': 6})


def sample_draw(
     prob,
     path='D:/todo/DeepLearning/dataset/TIMIT/TEST/DR1/MDAB0/SX49'):
    fig = plt.figure(figsize=(14, 24))
    draw_phoneme(fig, prob)
    draw_wav(fig, path)
    plt.savefig('alignment.png')
    plt.clf()


if __name__ == '__main__':
    phonemes.initialize()
    fig = plt.figure(figsize=(14, 24))
    draw_wav(fig)
