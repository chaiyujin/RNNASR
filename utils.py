import sys, time


def process_bar(x, total, length=50):
    x += 1
    line = '['
    step = length / total
    for _ in range(int(x * step)):
        line += '='
    if int(x * step) < length:
        line += '>'
    for _ in range(length - int(x * step) - 1):
        line += ' '
    line += ']'
    sys.stdout.write(line + ' ' + str(int(x * 100 / total)) + '%\r')
    sys.stdout.flush()


if __name__ == '__main__':
    for i in range(100):
        process_bar(i, 100)
        time.sleep(.1)
