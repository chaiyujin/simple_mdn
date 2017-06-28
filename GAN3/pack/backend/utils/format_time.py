def format_ms(ms):
    ms = int(ms)
    sec = int(ms / 1000)
    ms %= 1000
    min = int(sec / 60)
    sec %= 60
    hour = int(min / 60)
    min %= 60
    day = int(hour / 24)
    hour %= 24

    out = ''
    if day > 0:
        out += str(day) + 'd '
    if hour > 0 or len(out) > 0:
        out += str(hour) + 'h '
    if min > 0 or len(out) > 0:
        out += str(min) + 'm '
    if sec > 0 or len(out) > 0:
        out += str(sec) + 's '
    out += str(ms) + 'ms'
    return out


def format_sec(seconds):
    return format_ms(seconds * 1000)
