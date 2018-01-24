import datetime

def current_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')