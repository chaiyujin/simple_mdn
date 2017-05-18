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


def log(level='', title='', content='\n'):
    if len(title) > 0:
        title = '[' + title + '] '
    if level == 'warning' or level == 'error':
        title = color(title, 'red')
    elif level == 'log':
        title = color(title, 'yellow')
    elif level == 'info':
        title = color(title, 'green')
    else:
        pass
    sys.stdout.write(title + content)
    sys.stdout.flush()
