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
    return line + ' ' + str(int(x * 100 / total)) + '%'
