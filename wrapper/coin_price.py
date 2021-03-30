import pandas as pd
from urllib.request import urlopen
from datetime import datetime as dt
import json
import re
from wrapper import common

class CoinPrice:

    def bithumb_current_price(self, coin_cd='BTC'):

        url = 'https://api.bithumb.com/public/ticker/' + coin_cd

        source = urlopen(url).read()
        js = json.loads(source)

        current_price = js['data']['closing_price']
        current_price = float(current_price)

        bid_price = js['data']['buy_price']
        bid_price = float(bid_price)

        ask_price = js['data']['sell_price']
        ask_price = float(ask_price)

        pt = js['data']['date']
        pt = int(pt) / 1000
        present_time = dt.fromtimestamp(pt)
        present_time = common.utc_kst(present_time).strftime('%Y-%m-%d %H:%M:%S')

        return present_time, current_price, bid_price, ask_price

    def upbit_current_price(self, coin_cd='BTC'):

        url = 'https://api.upbit.com/v1/ticker?markets=KRW-' + coin_cd

        source = urlopen(url).read()
        js = json.loads(source)[0]

        current_price = js['trade_price']
        current_price = float(current_price)

        pt = js['trade_timestamp']
        pt = int(pt) / 1000
        present_time = dt.fromtimestamp(pt).strftime('%Y-%m-%d %H:%M:%S')

        return present_time, current_price

    def bithumb_historical_price(self, coin_cd='BTC', freq='M'):

        if freq == 'M':
            frequency = '01M'
        elif freq == 'H':
            frequency = '01H'
        elif freq == 'D':
            frequency = '24H'
        else:
            frequency = freq

        historical_prices_url = 'https://www.bithumb.com/resources/chart/' + coin_cd + '_xcoinTrade_' + frequency + '.json?symbol=' + coin_cd + '&resolution=0.5'
        source = urlopen(historical_prices_url).read()
        data = source.decode('utf-8')

        data = data.replace('[[', '')
        data = data.replace(']]', '')
        data = data.split('],[')

        hp_lst = list()
        for d in data:
            sub_d = list()
            for sd in d.split(','):
                sub_d.append(re.sub('[^\d.]', '', sd))
            hp_lst.append(sub_d)

        historical_data = dict()
        for d in hp_lst:
            pt = int(d[0]) / 1000
            present_time = dt.fromtimestamp(pt)
            d[0] = common.utc_kst(present_time).strftime('%Y-%m-%d %H:%M:%S')
            d[1] = float(d[1])  # 시가
            d[2] = float(d[2])  # 종가
            d[3] = float(d[3])  # 고가
            d[4] = float(d[4])  # 저가
            d[5] = round(float(d[5]), 4)
            historical_data[d[0]] = (d[2], d[1], d[3], d[4], d[5])

        historical_df = pd.DataFrame.from_dict(historical_data, orient='index', \
                                               columns=['close', 'open', 'high', 'low', 'volume'])
        # historical_df.sort_index(ascending=False, inplace=True)
        return historical_df