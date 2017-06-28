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


file_list = []


def add_log_file(path):
    file_list.append(open(path, 'a+'))


def log_file(level='', content='\n'):
    global file_list
    for file in file_list:
        # print(content)
        if len(level) > 0:
            level = '[' + level + '] '
        file.write(level + content)
        file.flush()


def close_log_files():
    for file in file_list:
        file.close()
