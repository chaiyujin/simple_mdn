import sys


def color(string, style):
    if style == 'yellow':
        string = '\033[01;33m' + string + '\033[0m'
    elif style == 'red':
        string = '\033[01;31m' + string + '\033[0m'
    elif style == 'green':
        string = '\033[01;32m' + string + '\033[0m'
    else:
        pass
    return string


def log(level, string):
    if level == 'warning' or level == 'error':
        string = color(string, 'red')
    elif level == 'log':
        string = color(string, 'yellow')
    elif level == 'info':
        string = color(string, 'green')
    else:
        pass
    sys.stdout.write(string)
    sys.stdout.flush()
