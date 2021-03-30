import pandas as pd
import numpy as np
from wrapper import common
import datetime as dt
import  pandas as pd
import  numpy as np
from pykrx import stock

class Trade:

    def BB_trade(self, df, base_date, book, cd, n, sigma):
        ##센터라인
        df['center'] = df[cd].rolling(n).mean()
        df['ub'] = df['center'] + sigma * df[cd].rolling(n).std()
        df['lb'] = df['center'] - sigma * df[cd].rolling(n).std()
        sample = df[base_date:].copy()
        for i in sample.index:
            price = sample.loc[i, cd]
            if price > sample.loc[i, 'ub']:  ## 어퍼바운드 위면
                book.loc[i, 't '+cd] = '' ##일단 놔두고
            elif sample.loc[i, 'ub'] >= price and price >= sample.loc[i, 'lb']: #사이에 있을떄
                if book.shift(1).loc[i, 't '+cd] == 'buy' or book.shift(1).loc[i, 't '+cd] == 'ready': ##이미 매수상태면
                    book.loc[i, 't '+cd] = 'buy'  ##그대로 유지
                else:                          ###빈손상태면
                    book.loc[i, 't '+cd] = ''  ##그대로 유지, 일단 놔둬
            elif sample.loc[i, 'lb'] > price: #가격이 아래 바운더리 뚫으면
                if book.shift(1).loc[i, 't '+cd] == 'buy':
                    book.loc[i, 't '+cd] = 'buy'  #매수
                else:
                    book.loc[i, 't ' + cd] = 'ready'
        status = ''
        for i in book.index:
            if book.loc[i, 't '+cd] == 'buy': #매수상태면
                if book.shift(1).loc[i, 't '+cd] == 'buy':
                    status = 'll'
                elif book.shift(1).loc[i, 't '+cd] == '':
                    status = 'zl'
                else:
                    status = 'zl'
            elif book.loc[i, 't '+cd] == '':
                if book.shift(1).loc[i, 't '+cd] == 'buy':
                    status = 'lz'
                elif book.shift(1).loc[i, 't '+cd] == '':
                    status = 'zz'
                else:
                    status = 'zz'
            else:
                status = 'zz'

            book.loc[i, 'p '+cd] = status




    def trade(self, df, book, cd, buy, sell):
        for i in df.index:
            price = df.loc[i, cd]
            if price < buy:
                book.loc[i, 't '+cd] = 'buy'
            elif price > sell:
                book.loc[i, 't '+cd] = '' ##포지션 청산기록
            else: ##가격이 구간 사이면
                if book.shift(1).loc[i, 't '+cd] == 'buy': #이미 매수상태이면
                    book.loc[i, 't '+cd] = 'buy'



    def rsi(df, period):

        for p in df.iloc[period:].index:
            d, ad, u, au = 0, 0., 0, 0.
            for i in range(period):
                diff = df.shift(i).loc[p, 'diff']  ## i일 전의 diff값을 읽어와서
                if diff >= 0:  # 양수이면
                    u += 1  # count 해주고
                    au += diff  ##누적으로 더해줘
                elif diff < 0:  # 음수면
                    d += 1  # count 해주고
                    ad -= diff  ## 누적으로 빼줘( 음수니까 마이너스해줘야 플러스로 돌아오지)
            if (au+ad) != 0:
                rsi = (au / (au + ad)) * 100
            else:
                rsi = 50

            df.loc[p, 'RSI'+str(period)] = rsi


    def trend_trading(self, sample, book, cd, factor):
        for i in sample.index:
            if sample.loc[i, factor] >= 0:  ##상승추세면
                book.loc[i,'t '+cd] = 'buy' # 사
            elif sample.loc[i, factor] < 0:
                book.loc[i,'t '+cd] = ''  ## 털어
            return (book)

    def date_format(self, d=''):
        if d != '':
            this_date = pd.to_datetime(d).date()
        else:
            this_date = pd.Timestamp.today().date()  # 오늘 날짜를 지정
        return (this_date)

    def check_base_date(self, prices_df, d):
        d = pd.to_datetime(d)
        prices_df.index = pd.to_datetime(prices_df.index)
        if d in pd.to_datetime(prices_df.index):
            return (d)
        else:
            nd = self.next_date(d)
            d = self.check_base_date(prices_df, nd)
            return (d)

    def next_date(self, d):
        d = d + pd.DateOffset(1)
        # d += dt.timedelta(1)
        return (d)

    def standardize(self, prices_df, base_date, codes):
        # codes = prices_df.columns.values
        for c in codes:
            std = prices_df[c] / prices_df.loc[base_date][c] * 100
            prices_df[c + ' idx'] = round(std, 2)
        return (prices_df)

    def sampling(self, prices_df, s_date, s_codes):
        sample = pd.DataFrame()
        sample = prices_df.loc[s_date:][s_codes].copy()
        return (sample)

    def create_trade_book(self, sample, s_cd):
        self.book = pd.DataFrame()
        self.book[s_cd] = sample[s_cd]
        # book['trade'] = ''
        if type(s_cd) == str:
            cds = []
            cds.append(s_cd)
        else:
            cds = s_cd
        for c in cds:
            self.book['t ' + c] = ''
            self.book['p ' + c] = ''
        return(self.book)

    def position(self, book, s_cd):
        if type(s_cd) == str:
            cds = []
            cds.append(s_cd)
        else:
            cds = s_cd
        for c in cds:
            status = ''
            for i in book.index:
                if book.loc[i, 't ' + c] == 'buy':
                    if book.shift(1).loc[i, 't ' + c] == 'buy':
                        status = 'll'
                    elif book.shift(1).loc[i, 't ' + c] == '':
                        status = 'zl'
                    elif book.shift(1).loc[i, 't ' + c] == 'sell':
                        status = 'sl'
                    else:
                        status = 'zl'
                elif book.loc[i, 't ' + c] == 'sell':
                    if book.shift(1).loc[i, 't ' + c] == 'buy':
                        status = 'ls'
                    elif book.shift(1).loc[i, 't ' + c] == '':
                        status = 'zs'
                    elif book.shift(1).loc[i, 't ' + c] == 'sell':
                        status = 'ss'
                    else:
                        status = 'zs'
                elif book.loc[i, 't ' + c] == '':
                    if book.shift(1).loc[i, 't ' + c] == 'buy':
                        status = 'lz'
                    elif book.shift(1).loc[i, 't ' + c] == '':
                        status = 'zz'
                    elif book.shift(1).loc[i, 't ' + c] == 'sell':
                        status = 'sz'
                    else:
                        status = 'zz'
                else:
                    status = 'zz'
                book.loc[i, 'p ' + c] = status
        return (book)

    def position_strategy(self, book, s_cd, last_date):

        if type(s_cd) == str:
            cds = []
            cds.append(s_cd)
        else:
            cds = s_cd

        strategy = ''
        for c in cds:
            i = book.index[-1]
            if book.loc[i, 'p ' + c] == 'lz' or book.loc[i, 'p ' + c] == 'sz' or book.loc[i, 'p ' + c] == 'zz':
                strategy = 'nothing'
            elif book.loc[i, 'p ' + c] == 'll' or book.loc[i, 'p ' + c] == 'sl' or book.loc[i, 'p ' + c] == 'zl':
                strategy += 'long ' + c
            elif book.loc[i, 'p ' + c] == 'ls' or book.loc[i, 'p ' + c] == 'ss' or book.loc[i, 'p ' + c] == 'zs':
                strategy += 'short ' + c
        print('As of', last_date, 'your model portfolio', cds, 'needs to be composed of', strategy)
        return (strategy)

    # 수익률
    def returns(self, book, s_cd, display=False, report_name='', report={}, fee=0.0):
        # 손익 계산
        cds = common.str_list(s_cd)

        rtn = 1.0
        book['return'] = 1
        no_trades = 0
        no_win = 0

        for c in cds:
            buy = 0.0
            sell = 0.0
            for i in book.index:

                if book.loc[i, 'p ' + c] == 'zl' or book.loc[i, 'p ' + c] == 'sl':  # long 진입
                    buy = book.loc[i, c]
                    if fee > 0.0:
                        buy = round(buy * (1 + fee), 3)
                    if display:
                        print(i, 'long ' + c, buy)

                elif book.loc[i, 'p ' + c] == 'lz' or book.loc[i, 'p ' + c] == 'ls':  # long 청산
                    sell = book.loc[i, c]
                    if fee > 0.0:
                        sell = round(sell * (1 - fee), 3)
                    # 손익 계산
                    rtn = sell / buy
                    book.loc[i, 'return'] = rtn
                    no_trades += 1
                    if rtn > 1:
                        no_win += 1
                    if display:
                        print(i, 'long ' + c, buy, ' | unwind long ' + c, sell,
                              ' | return: %0.2f' % (round(rtn - 1, 4) * 100))

                elif book.loc[i, 'p ' + c] == 'zs' or book.loc[i, 'p ' + c] == 'ls':  # short 진입
                    sell = book.loc[i, c]
                    if fee > 0.0:
                        sell = sell * (1 - fee)
                    if display:
                        print(i, 'short ' + c, sell)
                elif book.loc[i, 'p ' + c] == 'sz' or book.loc[i, 'p ' + c] == 'sl':  # short 청산
                    buy = book.loc[i, c]
                    if fee > 0.0:
                        buy = buy * (1 + fee)
                    # 손익 계산
                    rtn = buy / sell
                    book.loc[i, 'return'] = rtn
                    no_trades += 1
                    if rtn > 1:
                        no_win += 1
                    if display:
                        print(i, 'short ' + c, sell, ' | unwind short ' + c, buy,
                              ' | return: %0.2f' % (round(rtn - 1, 4) * 100))

            if book.loc[i, 't ' + c] == '' and book.loc[i, 'p ' + c] == '':  # zero position
                buy = 0.0
                sell = 0.0

        # Accumulated return
        acc_rtn = 1.0
        for i in book.index:
            rtn = book.loc[i, 'return']
            acc_rtn = acc_rtn * rtn
            book.loc[i, 'acc return'] = acc_rtn

        first_day = pd.to_datetime(book.index[0])
        last_day = pd.to_datetime(book.index[-1])
        total_days = (last_day - first_day).days
        annualizer = total_days / 365

        print(common.FontStyle.bg_white + 'Accumulated return:', round((acc_rtn - 1) * 100, 2), '%' + common.FontStyle.end_bg, \
              ' ( # of trade:', no_trades, ', # of win:', no_win, ', fee: %.2f' % (fee * 100), \
              '%,', 'period: %.2f' % annualizer, 'yr )')

        if no_trades > 0:
            avg_rtn = acc_rtn ** (1 / no_trades)
            prob_win = round((no_win / no_trades), 4)
        else:
            avg_rtn = 1.0
            prob_win = 0.0
        avg_rtn = round(avg_rtn, 4)

        bet = common.Bet()
        kelly_ratio = bet.kelly_formular(prob_win)
        kelly_ratio = round(kelly_ratio, 4)

        print('Avg return: %.2f' % ((avg_rtn - 1) * 100), end=' %')
        if prob_win > 0.5:
            print(common.FontStyle.orange, end='')
        print(', Prob. of win: %.2f' % (prob_win * 100), end=' %')
        if prob_win > 0.5:
            print(common.FontStyle.end_c, end='')
        print(', Kelly ratio: %.2f' % (kelly_ratio * 100), end=' %')

        mdd = round((book['return'].min()), 4)
        print(', MDD: %.2f' % ((mdd - 1) * 100), '%')

        if not report == {}:
            report['acc_rtn'] = round((acc_rtn) * 100, 2)
            report['no_trades'] = no_trades
            report['avg_rtn'] = round((avg_rtn * 100), 2)
            report['prob_win'] = round((prob_win * 100), 2)
            report['kelly_ratio'] = round((kelly_ratio * 100), 2)
            report['fee'] = round((fee * 100), 2)
            report['mdd'] = round((mdd * 100), 2)

        return round(acc_rtn, 4)

    # 벤치마크 수익
    def benchmark_return(self, book, s_cd, report_name='', report={}):
        # 벤치마크 수익률

        cds = common.str_list(s_cd)

        n = len(cds)
        rtn = dict()
        acc_rtn = float()
        for c in cds:
            rtn[c] = round((book[c].iloc[-1] - book[c].iloc[0]) / book[c].iloc[0] + 1, 4)
            acc_rtn += rtn[c] / n
        print('BM return:', round((acc_rtn - 1) * 100, 2), '%', rtn)
        if not report == {}:
            report['BM_rtn'] = (round((acc_rtn - 1) * 100, 2))
            report['BM_rtn_A'] = (round(rtn[cds[0]] * 100, 2))
            report['BM_rtn_B'] = (round(rtn[cds[1]] * 100, 2))

        return round(acc_rtn, 4)

    # 초과수익률
    def excess_return(self, fund_rtn, bm_rtn, report_name='', report={}):
        exs_rtn = fund_rtn - bm_rtn
        print('Excess return: %.2f' % (exs_rtn * 100), '%', \
              ' ( %.2f' % ((fund_rtn - 1) * 100), '- %.2f' % ((bm_rtn - 1) * 100), ')')
        if not report == {}:
            report['excess_rtn'] = round(exs_rtn * 100, 2)

        return exs_rtn

    def annualizer(self, book, cd):

        first_day = pd.to_datetime(book.index[0])
        last_day = pd.to_datetime(book.index[-1])
        total_days = (last_day - first_day).days
        if total_days < 1:
            total_days = 1
        annualizer = total_days / 365

        acc_return = book['acc return'][-1] - 1
        total_bm_return = (book[cd][-1] - book[cd][0]) / book[cd][0]
        if annualizer >= 1:
            annual_return = (acc_return + 1) ** (1 / annualizer) - 1
            annual_bm_return = (total_bm_return + 1) ** (1 / annualizer) - 1
        else:
            annual_return = acc_return * (1 / annualizer)
            annual_bm_return = total_bm_return * (1 / annualizer)
        annual_return = round(annual_return, 4)
        annual_bm_return = round(annual_bm_return, 4)

        print('Annual return: %.2f' % (annual_return * 100), end=' %')
        print(', Annual BM return: %.2f' % (annual_bm_return * 100), end=' %')
        print(', Annual excess return: %.2f' % ((annual_return - annual_bm_return) * 100), '%')

        return annual_return

    def returns_log(self, book, s_cd, display=False):
        # 손익 계산
        if type(s_cd) == str:
            cds = []
            cds.append(s_cd)
        else:
            cds = s_cd

        rtn = 0.0
        book['return'] = rtn

        for c in cds:
            buy = 0.0
            sell = 0.0
            for i in book.index:

                if book.loc[i, 'p ' + c] == 'zl' or book.loc[i, 'p ' + c] == 'sl':  # long 진입
                    buy = book.loc[i, c]
                    if display == True:
                        print(i.date(), 'long ' + c, buy)
                elif book.loc[i, 'p ' + c] == 'lz' or book.loc[i, 'p ' + c] == 'ls':  # long 청산
                    sell = book.loc[i, c]
                    # 손익 계산
                    rtn = np.log(sell / buy) * 100
                    # (sell - buy) / buy + 1
                    book.loc[i, 'return'] = rtn
                    if display == True:
                        print(i.date(), 'long ' + c, buy, ' | unwind long ' + c, sell, ' | return:', round(rtn, 4))

                elif book.loc[i, 'p ' + c] == 'zs' or book.loc[i, 'p ' + c] == 'ls':  # short 진입
                    sell = book.loc[i, c]
                    if display == True:
                        print(i.date(), 'short ' + c, sell)
                elif book.loc[i, 'p ' + c] == 'sz' or book.loc[i, 'p ' + c] == 'sl':  # short 청산
                    buy = book.loc[i, c]
                    # 손익 계산
                    rtn = np.log(sell / buy) * 100
                    book.loc[i, 'return'] = rtn
                    if display == True:
                        print(i.date(), 'short ' + c, sell, ' | unwind short ' + c, buy, ' | return:', round(rtn, 4))

            if book.loc[i, 't ' + c] == '' and book.loc[i, 'p ' + c] == '':  # zero position
                buy = 0.0
                sell = 0.0

        acc_rtn = 0.0
        for i in book.index:
            rtn = book.loc[i, 'return']
            acc_rtn = acc_rtn + rtn
            book.loc[i, 'acc return'] = acc_rtn

        print('Accunulated return :', round(acc_rtn, 2), '%')
        return (round(acc_rtn, 4))

    def benchmark_return_log(self, book, s_cd):
        # 벤치마크 수익률
        if type(s_cd) == str:
            cds = []
            cds.append(s_cd)
        else:
            cds = s_cd
        n = len(cds)
        rtn = dict()
        acc_rtn = float()
        for c in cds:
            rtn[c] = round(np.log(book[c].iloc[-1] / book[c].iloc[0]) * 100, 4)
            acc_rtn += rtn[c] / n
        print('BM return:', round(acc_rtn, 2), '%')
        print(rtn)
        return (round(acc_rtn, 4))

    def excess_return_log(self, fund_rtn, bm_rtn):
        exs_rtn = fund_rtn - bm_rtn
        print('Excess return:', round(exs_rtn, 2), '%')
        return (exs_rtn)


class SingleAsset(Trade):

    def bollinger_band(self, prices_df, s_cd, n, sigma):
        if type(s_cd) == str:
            cds = []
            cds.append(s_cd)
        else:
            cds = s_cd

        bb = pd.DataFrame()
        bb[cds[0]] = prices_df[cds[0]]
        bb['center'] = prices_df[cds[0]].rolling(n).mean()
        bb['ub'] = bb['center'] + sigma * prices_df[cds[0]].rolling(n).std()
        bb['lb'] = bb['center'] - sigma * prices_df[cds[0]].rolling(n).std()
        bb['band_size'] = bb['ub'] - bb['lb']
        return (bb)

    def tradings(self, sample, book, thd, s_cd, buy='in', short=False):
        if type(s_cd) == str:
            cds = []
            cds.append(s_cd)
        else:
            cds = s_cd

        for i in sample.index:
            price = sample.loc[i, cds[0]]

            if short == True:

                if price > sample.loc[i, 'ub']:
                    if book.shift(1).loc[i, 't ' + cds[0]] == 'sell':  # 이미 매수상태라면
                        book.loc[i, 't ' + cds[0]] = 'sell'  # 매수상태 유지
                    else:
                        if buy == 'in':  # 밴드 진입 시 매수
                            book.loc[i, 't ' + cds[0]] = ''  # 대기
                        else:
                            book.loc[i, 't ' + cds[0]] = 'sell'  # 매수

                elif sample.loc[i, 'ub'] >= price and price > sample.loc[i, 'center']:
                    if buy == 'out':
                        if book.shift(1).loc[i, 't ' + cds[0]] == 'sell':  # 숏 유지
                            book.loc[i, 't ' + cds[0]] = 'sell'
                        elif book.shift(1).loc[i, 't ' + cds[0]] == 'buy':  # 롱 청산
                            book.loc[i, 't ' + cds[0]] = ''
                        else:
                            book.loc[i, 't ' + cds[0]] = ''
                    else:
                        if book.shift(1).loc[i, 't ' + cds[0]] == 'sell' or book.shift(1).loc[i, 't ' + cds[0]] == '':
                            book.loc[i, 't ' + cds[0]] = 'sell'
                        elif book.shift(1).loc[i, 't ' + cds[0]] == 'buy':  # 롱 청산
                            book.loc[i, 't ' + cds[0]] = ''
                        else:
                            book.loc[i, 't ' + cds[0]] = ''

                elif sample.loc[i, 'center'] >= price and price > sample.loc[i, 'lb']:
                    if buy == 'out':
                        if book.shift(1).loc[i, 't ' + cds[0]] == 'sell':  # 숏 청산
                            book.loc[i, 't ' + cds[0]] = ''
                        elif book.shift(1).loc[i, 't ' + cds[0]] == 'buy':  # 롱 유지
                            book.loc[i, 't ' + cds[0]] = 'buy'
                        else:
                            book.loc[i, 't ' + cds[0]] = ''
                    else:
                        if book.shift(1).loc[i, 't ' + cds[0]] == 'sell':  # 숏 청산
                            book.loc[i, 't ' + cds[0]] = ''
                        elif book.shift(1).loc[i, 't ' + cds[0]] == 'buy' or book.shift(1).loc[i, 't ' + cds[0]] == '':
                            book.loc[i, 't ' + cds[0]] = 'buy'
                        else:
                            book.loc[i, 't ' + cds[0]] = ''

                elif sample.loc[i, 'lb'] >= price:
                    if book.shift(1).loc[i, 't ' + cds[0]] == 'buy':  # 이미 매수상태라면
                        book.loc[i, 't ' + cds[0]] = 'buy'  # 매수상태 유지
                    else:
                        if buy == 'in':  # 밴드 진입 시 매수
                            book.loc[i, 't ' + cds[0]] = ''  # 대기
                        else:
                            book.loc[i, 't ' + cds[0]] = 'buy'  # 매수

            else:

                if price > sample.loc[i, thd]:
                    book.loc[i, 't ' + cds[0]] = ''
                elif sample.loc[i, thd] >= price and price >= sample.loc[i, 'lb']:
                    if buy == 'in':
                        if book.shift(1).loc[i, 't ' + cds[0]] == 'buy' or book.shift(1).loc[
                            i, 't ' + cds[0]] == 'ready':
                            # 이미 매수상태 또는 Ready에서 넘어온 상태
                            book.loc[i, 't ' + cds[0]] = 'buy'  # trade : buy (매수상태 유지)
                        else:
                            book.loc[i, 't ' + cds[0]] = ''  # trade : clear (zero상태 유지)
                    else:
                        if book.shift(1).loc[i, 't ' + cds[0]] == 'buy':
                            book.loc[i, 't ' + cds[0]] = 'buy'
                        else:
                            book.loc[i, 't ' + cds[0]] = ''
                elif sample.loc[i, 'lb'] > price:
                    if buy == 'in':
                        if book.shift(1).loc[i, 't ' + cds[0]] == 'buy':
                            book.loc[i, 't ' + cds[0]] = 'buy'  # 이미 buy
                        else:
                            book.loc[i, 't ' + cds[0]] = 'ready'
                    else:
                        book.loc[i, 't ' + cds[0]] = 'buy'  # 매수상태 유지
        return (book)

    def trading_strategy(self, sample, thd, s_cd, last_date):
        if type(s_cd) == str:
            cds = []
            cds.append(s_cd)
        else:
            cds = s_cd
        i = sample.index[-1]
        if sample.loc[i, cds[0]] >= sample.loc[i, thd]:
            strategy = ''
        elif sample.loc[i, cds[0]] <= sample.loc[i, 'lb']:
            strategy = 'buy ' + cds[0]
        else:
            strategy = 'just wait'
        print('As of', last_date, 'this model suggests you to', strategy)
        return (strategy)


class MultiAsset(Trade):
    pass


class PairTrade(Trade):

    def regression(self, sample, s_codes):
        sample.dropna(inplace=True)
        from sklearn.linear_model import LinearRegression
        x = sample[s_codes[0]]
        y = sample[s_codes[1]]
        # 1개 컬럼 np.array로 변환
        x = np.array(x).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        # Linear Regression
        regr = LinearRegression()
        regr.fit(x, y)
        result = {'Slope': regr.coef_[0, 0], 'Intercept': regr.intercept_[0], 'R2': regr.score(x, y)}
        # result = {'Slope':regr.coef_, 'Intercept':regr.intercept_, 'R2':regr.score(x, y) }
        return (result)

    def compare_r2(self, prices_df, base_date, s_codes):
        comp_df = pd.DataFrame()
        s_df = self.sampling(prices_df, base_date, s_codes)
        s_df = s_df.dropna()
        n = len(s_codes)
        for i in range(0, n, 1):
            for j in range(i, n, 1):
                if i != j:
                    code_pairs = [s_codes[i], s_codes[j]]
                    regr = self.regression(s_df, code_pairs)
                    c_pair = s_codes[i] + ' vs. ' + s_codes[j]
                    # print(s_codes[i], '-', s_codes[j], ' : ', '{:,.2f}'.format(regr['R2']*100))
                    comp_df.loc[c_pair, 'R2'] = round(regr['R2'], 4) * 100
                    comp_df.loc[c_pair, 'Slope'] = round(regr['Slope'], 4)
                    comp_df.loc[c_pair, 'Correlation'] = s_df[code_pairs].corr(method='pearson', min_periods=1).iloc[
                        1, 0]
        comp_df.index.name = 'pair'
        comp_df = comp_df.sort_values(by='R2', ascending=False)
        return (comp_df)

    def expected_y(self, sample, regr, s_codes):
        sample[s_codes[1] + ' expected'] = sample[s_codes[0]] * regr['Slope'] + regr['Intercept']
        sample[s_codes[1] + ' spread'] = sample[s_codes[1]] - sample[s_codes[1] + ' expected']
        return (sample)

    def price_analyze(self, sample, thd, s_codes):
        for i in sample.index:
            threshold = float(thd * sample.loc[i, s_codes[1]])
            if sample.loc[i, s_codes[1] + ' spread'] > threshold:
                sample.loc[i, 'cheaper'] = s_codes[0]
            elif sample.loc[i, s_codes[1] + ' spread'] < -threshold:
                sample.loc[i, 'cheaper'] = s_codes[1]
            else:
                sample.loc[i, 'cheaper'] = 'E'
        print(sample.groupby('cheaper').count())
        return (sample)

    def tradings(self, sample, book, thd, s_codes, short=False):
        for i in sample.index:
            threshold = float(thd * sample.loc[i, s_codes[1]])
            if sample.loc[i, s_codes[1] + ' spread'] > threshold:
                book.loc[i, 't ' + s_codes[0]] = 'buy'
                if short == True:
                    book.loc[i, 't ' + s_codes[1]] = 'sell'
                else:
                    book.loc[i, 't ' + s_codes[1]] = ''
            elif threshold >= sample.loc[i, s_codes[1] + ' spread'] and sample.loc[i, s_codes[1] + ' spread'] >= 0:
                book.loc[i, 't ' + s_codes[0]] = ''
                book.loc[i, 't ' + s_codes[1]] = ''
            elif 0 > sample.loc[i, s_codes[1] + ' spread'] and sample.loc[i, s_codes[1] + ' spread'] >= -threshold:
                book.loc[i, 't ' + s_codes[0]] = ''
                book.loc[i, 't ' + s_codes[1]] = ''
            elif -threshold > sample.loc[i, s_codes[1] + ' spread']:
                if short == True:
                    book.loc[i, 't ' + s_codes[0]] = 'sell'
                else:
                    book.loc[i, 't ' + s_codes[0]] = ''
                book.loc[i, 't ' + s_codes[1]] = 'buy'
        return (book)

    def tradings_old(self, sample, book, thd, s_codes):
        for i in sample.index:
            threshold = float(thd * sample.loc[i, s_codes[1]])
            if sample.loc[i, s_codes[1] + ' spread'] > threshold:
                book.loc[i, 'trade'] = 'buy ' + s_codes[0]
            elif threshold >= sample.loc[i, s_codes[1] + ' spread'] and sample.loc[i, s_codes[1] + ' spread'] >= 0:
                book.loc[i, 'trade'] = ''
            elif 0 > sample.loc[i, s_codes[1] + ' spread'] and sample.loc[i, s_codes[1] + ' spread'] >= -threshold:
                book.loc[i, 'trade'] = ''
            elif -threshold > sample.loc[i, s_codes[1] + ' spread']:
                book.loc[i, 'trade'] = 'buy ' + s_codes[1]
        return (book)

    def trading_strategy(self, sample, thd, s_codes, last_date, short=False):
        i = sample.index[-1]
        threshold = float(thd * sample.loc[i, s_codes[1]])

        if sample.loc[i, s_codes[1] + ' spread'] > threshold:
            strategy = 'buy ' + s_codes[0]
            if short == True:
                strategy += ' & sell ' + s_codes[1]
            else:
                strategy += ' & clear ' + s_codes[1]
        elif threshold >= sample.loc[i, s_codes[1] + ' spread'] and sample.loc[i, s_codes[1] + ' spread'] >= 0:
            strategy = 'clear ' + s_codes[0]
            strategy += ' & clear ' + s_codes[1]
        elif 0 > sample.loc[i, s_codes[1] + ' spread'] and sample.loc[i, s_codes[1] + ' spread'] >= -threshold:
            strategy = 'clear ' + s_codes[0]
            strategy += ' & clear ' + s_codes[1]
        elif -threshold > sample.loc[i, s_codes[1] + ' spread']:
            strategy = 'buy ' + s_codes[1]
            if short == True:
                strategy += ' & sell ' + s_codes[0]
            else:
                strategy += ' & clear ' + s_codes[0]

        print('As of', last_date, 'this model suggests you to', strategy)
        return (strategy)

'''
class FuturesTradeOnValue(PairTrade):

    def expected_y(self, sample, s_codes, r, d, T):
        from finterstellar import Valuation
        vu = Valuation()
        for i in sample.index:
            sample.loc[i, s_codes[1] + ' expected'] = vu.futures_price(sample.loc[i, s_codes[0]], r, d, i, T)
        sample[s_codes[1] + ' spread'] = sample[s_codes[1]] - sample[s_codes[1] + ' expected']
        return (sample)

    def intraday_expected_y(self, sample, s_codes, r, d, t, T):
        from finterstellar import Valuation
        vu = Valuation()
        for i in sample.index:
            sample.loc[i, s_codes[1] + ' expected'] = vu.futures_price(sample.loc[i, s_codes[0]], r, d, t, T)
        sample[s_codes[1] + ' spread'] = sample[s_codes[1]] - sample[s_codes[1] + ' expected']
        return (sample)

    def price_analyze(self, sample, thd, s_codes):
        for i in sample.index:
            threshold = float(thd * sample.loc[i, s_codes[1]])
            if sample.loc[i, s_codes[1] + ' spread'] > 0:
                sample.loc[i, 'cheaper'] = s_codes[0]
            elif sample.loc[i, s_codes[1] + ' spread'] < -threshold:
                sample.loc[i, 'cheaper'] = s_codes[1]
            else:
                sample.loc[i, 'cheaper'] = 'E'
        print(sample.groupby('cheaper').count())
        return (sample)

    def tradings(self, sample, book, thd, s_codes):
        for i in sample.index:
            threshold = float(thd * sample.loc[i, s_codes[1]])
            if sample.loc[i, s_codes[1] + ' spread'] > threshold:
                book.loc[i, 't ' + s_codes[1]] = 'sell'
            elif threshold >= sample.loc[i, s_codes[1] + ' spread'] and sample.loc[i, s_codes[1] + ' spread'] >= 0:
                book.loc[i, 't ' + s_codes[1]] = 'sell'
            elif 0 > sample.loc[i, s_codes[1] + ' spread'] and sample.loc[i, s_codes[1] + ' spread'] >= -threshold:
                book.loc[i, 't ' + s_codes[1]] = ''
            elif -threshold > sample.loc[i, s_codes[1] + ' spread']:
                book.loc[i, 't ' + s_codes[1]] = 'buy'
        return (book)

    def trading_strategy(self, sample, thd, s_codes, last_date):
        i = sample.index[-1]
        threshold = float(thd * sample.loc[i, s_codes[1]])

        if sample.loc[i, s_codes[1] + ' spread'] > threshold:
            strategy = 'sell ' + s_codes[1]
        elif threshold >= sample.loc[i, s_codes[1] + ' spread'] and sample.loc[i, s_codes[1] + ' spread'] >= 0:
            strategy = 'sell ' + s_codes[1]
        elif 0 > sample.loc[i, s_codes[1] + ' spread'] and sample.loc[i, s_codes[1] + ' spread'] >= -threshold:
            strategy = 'do nothing'
        elif -threshold > sample.loc[i, s_codes[1] + ' spread']:
            strategy = 'buy ' + s_codes[1]

        print('As of', last_date, 'this model suggests you to', strategy)
        return (strategy)


class FuturesTradeOnBasis(Trade):

    def basis_calculate(self, df, pair):
        basis = df[pair[1]] - df[pair[0]]
        df['basis'] = basis
        return (df)

    def price_analyze(self, sample, thd, s_codes):
        for i in sample.index:
            if sample.loc[i, 'basis'] > thd:
                sample.loc[i, 'cheaper'] = s_codes[0]
            elif sample.loc[i, 'basis'] < 0:
                sample.loc[i, 'cheaper'] = s_codes[1]
            else:
                sample.loc[i, 'cheaper'] = 'E'
        print(sample.groupby('cheaper').count())
        return (sample)

    def tradings(self, sample, book, thd, s_codes):
        for i in sample.index:
            if sample.loc[i, 'basis'] > thd:
                book.loc[i, 't ' + s_codes[1]] = 'sell'
            elif thd >= sample.loc[i, 'basis'] and sample.loc[i, 'basis'] >= 0:
                book.loc[i, 't ' + s_codes[1]] = ''
            elif 0 > sample.loc[i, 'basis']:
                book.loc[i, 't ' + s_codes[1]] = 'buy'
        return (book)
'''
'''
    def trading_strategy(self, sample, thd, s_codes, last_date):
        i = sample.index[-1]

        if sample.loc[i, 'basis'] > thd:
            strategy = 'sell ' + s_codes[1]
        elif thd >= sample.loc[i, 'basis'] and sample.loc[i, 'basis'] >= 0:
            strategy = 'do nothing'
        elif 0 > sample.loc[i, 'basis']:
            strategy = 'buy ' + s_codes[1]

        print('As of', last_date, 'this model suggests you to', strategy)
        return (strategy)
'''