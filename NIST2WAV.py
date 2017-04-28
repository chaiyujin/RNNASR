import os


def convert_nist(root_dir='.\\TIMIT\\'):
    for parent, dirs, files in os.walk(root_dir):
        for file in files:
            name, ext = os.path.splitext(os.path.join(parent, file))
            if ext == '.NIST':
                print(name)
                os.system(
                    'sndfile-convert.exe ' + os.path.join(parent, file) +
                    ' ' + name + '.WAV')


if __name__ == '__main__':
    convert_nist()
