import pandas as pd
import sys
from datetime import datetime
from pytz import timezone, utc


def str_list(s_cd):
    cds = []
    if type(s_cd) == str:
        cds = []
        cds.append(s_cd)
    else:
        cds = s_cd
    return cds


def today_yymmdd():
    d = pd.Timestamp.today().date().strftime('%y%m%d')
    return d


def present_date():
    d = pd.Timestamp.today().strftime('%Y-%m-%d')
    return d


def present_time():
    d = pd.Timestamp.today().strftime('%H:%M:%S')
    return d


def date_format(d=''):
    if d != '':
        this_date = pd.to_datetime(d).date()
    else:
        this_date = pd.Timestamp.today().date()  # 오늘 날짜를 지정
    return this_date


def check_base_date(prices_df, d):
    d = pd.to_datetime(d)
    prices_df.index = pd.to_datetime(prices_df.index)
    if d in pd.to_datetime(prices_df.index):
        return d
    else:
        nd = next_date(d)
        d = check_base_date(prices_df, nd)
        return d


def next_date(d):
    d = d + pd.DateOffset(1)
    return d


def str_number(s):
    if type(s) == str:
        n = s.replace(',', '')
        n = float(n)
    return n


def str_number_array(s):
    lst = []
    for i in s:
        n = str_number(i)
        lst.append(n)
    return lst


def progress_bar(current_val, total_val, display_value, bar_length=30):
    progress = current_val / total_val
    done = '#' * round(progress * bar_length)
    undone = '.' * (bar_length - len(done))
    sys.stdout.write('\r[{0}] {1}% ({2}/{3}) {4}'. \
                     format(done + undone, int(round(progress * 100)), \
                            current_val, total_val, display_value))
    sys.stdout.flush()


def utc_kst(t):
    KST = timezone('Asia/Seoul')
    if not type(t) == datetime:
        t = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
    time_delta = round((datetime.now() - datetime.utcnow()).seconds / 3600)
    if time_delta == 0 or time_delta == 24:
        t = utc.localize(t).astimezone(KST)
    return t


class Bet:

    def kelly_formular(self, p, b=1):
        f = (p * (b + 1) - 1) / b
        if f < 0:
            f = 0
        return f


class FontStyle:
    header = '\033[95m'
    end_c = '\033[0m'
    blue = '\033[94m'
    green = '\033[92m'
    orange = '\033[93m'
    red = '\033[91m'
    bold = '\033[1m'
    underline = '\033[4m'
    end_bg = '\x1b[0m'
    bg_green = '\x1b[6;30;42m'
    bg_blue = '\x1b[1;30;44m'
    bg_white = '\x1b[1;30;47m'
    bg_red = '\x1b[1;37;41m'