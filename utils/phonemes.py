inited = False
g_phns_list = []
g_phns_map = {}
g_phonemes_number = 39


def initialize(path='./phonemes'):
    global inited
    global g_phns_list
    global g_phns_map

    inited = True
    raw_phns, _ = raw_phonemes(path)
    for phn in raw_phns:
        if convert_phoneme(phn) is not None:
            g_phns_map[convert_phoneme(phn)] = 1

    g_phns_list = []
    for k in g_phns_map:
        g_phns_list.append(k)
    g_phns_list.sort()

    for i, phn in enumerate(g_phns_list):
        g_phns_map[phn] = i


def get_phoneme_id(phn):
    global inited
    assert(inited)
    if convert_phoneme(phn) is None:
        return None
    return g_phns_map[convert_phoneme(phn)]


def convert_phoneme(phn):
    if phn == 'aa' or phn == 'ao':
        return 'aa'
    elif phn == 'ah' or phn == 'ax' or phn == 'ax-h':
        return 'ah'
    elif phn == 'er' or phn == 'axr':
        return 'er'
    elif phn == 'hh' or phn == 'hv':
        return 'hh'
    elif phn == 'ih' or phn == 'ix':
        return 'ih'
    elif phn == 'l' or phn == 'el':
        return 'l'
    elif phn == 'm' or phn == 'em':
        return 'm'
    elif phn == 'n' or phn == 'en' or phn == 'nx':
        return 'n'
    elif phn == 'ng' or phn == 'eng':
        return 'ng'
    elif phn == 'sh' or phn == 'zh':
        return 'sh'
    elif phn == 'pcl' or phn == 'tcl' or phn == 'kcl' or\
            phn == 'bcl' or phn == 'dcl' or phn == 'gcl' or\
            phn == 'h#' or phn == 'pau' or phn == 'epi':
        return 'sil'
    elif phn == 'uw' or phn == 'ux':
        return 'uw'
    elif phn == 'q':
        return None
    else:
        return phn


def raw_phonemes(path='./phonemes'):
    phoneme_list = []
    phoneme_map = {}
    with open(path) as file:
        idx = 0
        for line in file:
            phoneme_list.append(line.strip().split(' ')[0])
            phoneme_map[line.strip().split(' ')[0]] = idx
            idx += 1
    return phoneme_list, phoneme_map


if __name__ == '__main__':
    initialize()

    print(len(g_phns_list))
    for i, phn in enumerate(g_phns_list):
        print(phn + ' ' + str(g_phns_map[phn]) + ' ' + str(i))
