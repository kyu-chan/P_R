import pandas as pd
import datetime as dt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from wrapper import common


pd.plotting.deregister_matplotlib_converters()

font = 'NanumSquareRound, AppleGothic, Malgun Gothic, DejaVu Sans'


class Visualize:
    today = '(' + pd.to_datetime('today').date().strftime("%y%m%d") + ') '
    today_str = pd.to_datetime('today').date().strftime("%Y%m%d")

    def __init__(self):
        plt.style.use('fivethirtyeight')
        plt.rcParams['font.family'] = font
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['axes.grid'] = True
        plt.rcParams['lines.linewidth'] = 1.5
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.alpha'] = 0.7
        plt.rcParams['lines.antialiased'] = True
        plt.rcParams['figure.figsize'] = [15.0, 7.0]
        plt.rcParams['savefig.dpi'] = 96
        plt.rcParams['font.size'] = 12
        plt.rcParams['legend.fontsize'] = 'medium'
        plt.rcParams['figure.titlesize'] = 'medium'

    def price_view(self, df, b_date, cd, size=(15, 7), make_file=False):

        cds = common.str_list(cd)

        fig, ax = plt.subplots(figsize=size)
        x = df.loc[b_date:].index

        for c in cds:
            plt.plot(x, df.loc[b_date:, c], label=c)

        plt.legend()

        if make_file:
            plt.savefig('./image/' + self.today + cds[0] + ' price_view.png', bbox_inches='tight')

    def index_view(self, df, b_date, cd, size=(15, 7), make_file=False):

        if isinstance(df.index[0], dt.date):
            b_date = common.check_base_date(df, b_date)

        fig, ax = plt.subplots(figsize= size)
#        fig, ax = plt.subplots(figsize=size)

        x = df.loc[b_date:].index

        cds = common.str_list(cd)

        for c in cds:
            plt.plot(x, df.loc[b_date:, c] / df.loc[b_date, c] * 100, label=c)

        plt.legend()
        plt.show()
        if make_file:
            plt.savefig('./image/' + self.today + cds[0] + ' index_view.png', bbox_inches='tight')

    def complex_view(self, df, b_date, cd_a, cd_b, size=(15, 7), make_file=False):

        cds_a = common.str_list(cd_a)
        cds_b = common.str_list(cd_b)

        fig, ax1 = plt.subplots(figsize=size)
        x = df.loc[b_date:].index

        i = 1
        for c in cds_a:
            if i == 1:
                ax1.plot(x, df.loc[b_date:, c], color='C' + str(i), lw=3, label=c)
            else:
                ax1.plot(x, df.loc[b_date:, c], color='C' + str(i), label=c)
            i += 1

        if cds_b:
            ax2 = ax1.twinx()

            i = 6
            for c in cds_b:
                ax2.fill_between(x, df.loc[b_date:, c], 0, facecolor='C' + str(i), alpha=0.3)
                ax1.plot(np.nan, color='C' + str(i), label=c)
                i += 1

        ax1.legend(loc=0)

        if make_file:
            plt.savefig('./image/' + self.today + cds_a[0] + ' complex_view.png', bbox_inches='tight')

    def multi_line_view(self, df, b_date, cd_a, cd_b, size=(15, 7), make_file=False):

        cds_a = common.str_list(cd_a)
        cds_b = common.str_list(cd_b)

        fig, ax1 = plt.subplots(figsize=size)
        x = df.loc[b_date:].index

        i = 1
        for c in cds_a:
            if i == 1:
                ax1.plot(x, df.loc[b_date:, c], color='C' + str(i), lw=3, label=c)
                pass
            else:
                ax1.plot(x, df.loc[b_date:, c], color='C' + str(i), label=c)
            i += 1

        if cds_b:
            ax2 = ax1.twinx()

            i = 6
            for c in cds_b:
                ax2.plot(x, df.loc[b_date:, c], color='C' + str(i), label=c, alpha=0.7)
                ax1.plot(np.nan, color='C' + str(i), label=c)
                i += 1

        ax1.legend(loc=0)

        if make_file:
            plt.savefig('./image/' + self.today + cds_a[0] + ' multi_line_view.png', bbox_inches='tight')

    def position_view(self, df, cd, size=(15, 1), make_file=False, file_name=''):
        cds = common.str_list(cd)

        fig, ax = plt.subplots(figsize=size)
        x = df.index

        for c in cds:
            df['ps' + c] = 0
            df.loc[df['p ' + c] == 'll', ['ps' + c]] = 1
            df.loc[df['p ' + c] == 'sl', ['ps' + c]] = 1
            df.loc[df['p ' + c] == 'zl', ['ps' + c]] = 1
            df.loc[df['p ' + c] == 'ls', ['ps' + c]] = -1
            df.loc[df['p ' + c] == 'ss', ['ps' + c]] = -1
            df.loc[df['p ' + c] == 'zs', ['ps' + c]] = -1
            plt.fill_between(x, df['ps' + c], 0, label=c)
        plt.yticks([-1, 0, 1], ["Short", "Zero", "Long"])
        plt.legend()

        if make_file:
            f_name = file_name + '_position_view.png'
            plt.savefig('./image/' + f_name, bbox_inches='tight')

    def position_view_bar(self, df, cd, size=(15, 1), make_file=False):

        cds = common.str_list(cd)

        fig, ax = plt.subplots(figsize=size)
        x = df.index

        x_ticks = self.time_serial(df)
        plt.xticks(x_ticks[0], x_ticks[1])
        plt.autoscale(True, axis='x')

        for c in cds:
            df['ps' + c] = 0
            df.loc[df['p ' + c] == 'll', ['ps' + c]] = 1
            df.loc[df['p ' + c] == 'sl', ['ps' + c]] = 1
            df.loc[df['p ' + c] == 'zl', ['ps' + c]] = 1
            df.loc[df['p ' + c] == 'ls', ['ps' + c]] = -1
            df.loc[df['p ' + c] == 'ss', ['ps' + c]] = -1
            df.loc[df['p ' + c] == 'zs', ['ps' + c]] = -1
            plt.bar(range(x.size), df['ps' + c], width=1, label=c)
        plt.yticks([-1, 0, 1], ["Short", "Zero", "Long"])
        plt.legend()

        if make_file:
            plt.savefig('./image/' + self.today + cds[0] + ' position_view.png', bbox_inches='tight')

    def pair_trend_index_view(self, df, trd, cd, size=(15, 7), make_file=False, file_name=''):

        fig, ax1 = plt.subplots(figsize=size)
        x = df.index

        ax1.fill_between(x, df[cd[1] + ' expected'] * (1 + trd), df[cd[1] + ' expected'] * (1 - trd),
                         facecolor='sienna', alpha=0.2)
        ax1.plot(x, df[cd[1] + ' expected'], 'sienna', linestyle='--')
        ax1.plot(x, df[cd[1]], 'C1', lw=3)

        ax2 = ax1.twinx()

        ax2.plot(x, df[cd[0]], 'C0', alpha=0.7)
        ax1.plot(np.nan, 'C0', label=cd[0])

        ax1.legend(loc=0)

        if make_file:
            f_name = file_name + '_pair_trend_view.png'
            plt.savefig('./image/' + f_name, bbox_inches='tight')

        return ()

    def pair_trend_price_view(self, df, trd, cd, size=(15, 7), make_file=False):
        fig, ax = plt.subplots(figsize=size)
        x = df.index

        plt.fill_between(x, df[cd[1] + ' expected'] * (1 + trd), df[cd[1] + ' expected'] * (1 - trd),
                         facecolor='sienna', alpha=0.2)
        plt.plot(x, df[cd[1] + ' expected'], 'sienna', linestyle='--')
        plt.plot(x, df[cd[0]], 'C0')
        plt.plot(x, df[cd[1]], 'C1', lw=3)
        plt.legend()

        if make_file:
            plt.savefig('./image/' + self.today + cd[0] + ' pair_trend_price_view.png', bbox_inches='tight')

    def BB_trend_view(self, df, cd, size=(15, 7), make_file=False):
        cds = common.str_list(cd)

        fig, ax = plt.subplots(figsize=size)
        x = df.index

        plt.fill_between(x, df['lb'], df['ub'], facecolor='sienna', alpha=0.2)
        plt.plot(x, df['center'], color='sienna', linestyle='--', label='MA')
        plt.plot(x, df[cds[0]], color='C0', linestyle='-', lw=3)
        plt.legend()

        if make_file:
            plt.savefig('./image/' + self.today + cds[0] + ' bb_trend_view.png', bbox_inches='tight')

    def futures_basis_view(self, df, threshold, cd, size=(15, 7), make_file=False):
        cds = common.str_list(cd)

        fig, ax = plt.subplots(figsize=size)
        x = df.index

        plt.autoscale(True, axis='both')

        plt.fill_between(x, df[cds[0]], df[cds[0]] + df['basis'], facecolor='sienna', alpha=0.2)
        plt.plot(x, df[cds[0]], 'sienna', linestyle='--')
        plt.plot(x, df[cds[1]], 'C1', lw=3)
        plt.legend()

        if make_file:
            plt.savefig('./image/' + self.today + cds[0] + ' futures_basis_view.png', bbox_inches='tight')

    def value_at_expiry_view(self, x, make_file=False, size=(7, 7), **y):
        fig, ax = plt.subplots(figsize=size)
        plt.axhline(y=0, color='k', linewidth=1)  # x축
        s = pd.Series(0 for _ in range(len(x)))
        if len(y) > 1:
            for key, value in y.items():
                plt.plot(x, value, linestyle='--', linewidth=1, label=key)
                s = s + pd.Series(value)
            plt.plot(x, s, linewidth=3, color='red', label='Synthetic')
        else:
            for key, value in y.items():
                plt.plot(x, value, linewidth=3, color='red', label=key)
        step = (x.max() - x.min() + 1) / 4
        plt.yticks(np.arange(0 - step * 2, 0 + step * 3, step))
        plt.ylim(0 - step * 2, 0 + step * 2)
        plt.legend()

        if make_file:
            plt.savefig('./image/' + self.today + ' value_at_expiry_view.png', bbox_inches='tight')

    def square_one_to_one_view(self, x, make_file=False, size=(7, 7), **y):
        fig, ax = plt.subplots(figsize=size)
        plt.axhline(y=0, color='k', linewidth=1)  # x축
        s = pd.Series(0 for _ in range(len(x)))
        if len(y) > 1:
            for key, value in y.items():
                plt.plot(x, value, linestyle='--', linewidth=1, label=key)
                s = s + pd.Series(value)
            plt.plot(x, s, linewidth=3, color='red', label='Synthetic')
        else:
            for key, value in y.items():
                plt.plot(x, value, linewidth=3, color='red', label=key)
        step = (x.max() - x.min() + 1) / 4
        plt.yticks(np.arange(0 - step * 2, 0 + step * 3, step))
        plt.ylim(0 - step * 2, 0 + step * 2)
        plt.legend()

        if make_file:
            plt.savefig('./image/' + self.today + ' square_one_to_one_view.png', bbox_inches='tight')

    def square_free_plot_view(self, x, make_file=False, size=(7, 7), **y):

        fig, ax = plt.subplots(figsize=size)
        plt.axhline(y=0, color='k', linewidth=1)  # x축
        s = pd.Series(0 for _ in range(len(x)))
        if len(y) > 1:
            for key, value in y.items():
                plt.plot(x, value, linestyle='--', linewidth=1, label=key)
                s = s + pd.Series(value)
            plt.plot(x, s, linewidth=3, color='red', label='Synthetic')
        else:
            for key, value in y.items():
                plt.plot(x, value, linewidth=3, color='red', label=key)

        plt.legend()

        if make_file:
            plt.savefig('./image/' + Visualize.today + ' square_free_plot_view.png', bbox_inches='tight')

    def square_scatter_view(self, x, y, make_file=False, size=(7, 7)):

        fig, ax = plt.subplots(figsize=size)
        plt.axhline(y=0, color='k', linewidth=1)  # x축

        plt.scatter(x, y, linewidth=3, color='red')
        step = (x.max() - x.min() + 1) / 4

        plt.legend()

        if make_file:
            plt.savefig('./image/' + Visualize.today + ' square_free_plot_view.png', bbox_inches='tight')

    def time_serial(self, df):
        chart = pd.DataFrame()
        chart = df.copy()
        chart.reset_index(inplace=True)

        sequence = []
        xlabels = []

        if isinstance(chart.iloc[0, 0], dt.date):

            first = chart.iloc[0, 0]
            last = chart.iloc[-1, 0]

            delta = last - first
            if delta.days >= 730:
                time_series = pd.date_range(first, last, freq='YS')
            elif delta.days >= 365:
                time_series = pd.date_range(first, last, freq='QS')
            elif delta.days >= 180:
                time_series = pd.date_range(first, last, freq='2MS')
            elif delta.days >= 90:
                time_series = pd.date_range(first, last, freq='MS')
            elif delta.days >= 60:
                time_series = pd.date_range(first, last, freq='SMS')
            elif delta.days >= 30:
                time_series = pd.date_range(first, last, freq='5B')
            elif delta.days >= 10:
                time_series = pd.date_range(first, last, freq='2B')
            elif delta.days >= 5:
                time_series = pd.date_range(first, last, freq='D')
            else:
                time_series = chart.iloc[:, 0]

            sequence.append(first)
            if delta.days >= 180:
                xlabels.append(first.strftime('%y.%m.%d'))
            else:
                xlabels.append(first.strftime('%m.%d'))
            for d in time_series:
                d = common.check_base_date(df, d)
                s = chart[chart.iloc[:, 0] == d].iloc[0].tolist()
                sequence.append(s[0])
                l = d.strftime('%y.%m.%d')
                if delta.days >= 180:
                    l = d.strftime('%y.%m.%d')
                else:
                    l = d.strftime('%m.%d')
                xlabels.append(l)
            sequence.append(last)
            if delta.days >= 180:
                xlabels.append(last.strftime('%y.%m.%d'))
            else:
                xlabels.append(last.strftime('%m.%d'))

            if sequence[0] == sequence[1]:
                del sequence[0]
                del xlabels[0]
            if sequence[-1] == sequence[-2]:
                del sequence[-1]
                del xlabels[-1]

        return (sequence, xlabels)


'''
    intraday charting
'''


class VisualizeIntraday:
    today = '(' + pd.to_datetime('today').date().strftime("%y%m%d") + ') '

    def __init__(self):
        plt.style.use('fivethirtyeight')
        plt.rcParams['font.family'] = font
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['axes.grid'] = True
        plt.rcParams['lines.linewidth'] = 1.5
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.alpha'] = 0.7
        plt.rcParams['lines.antialiased'] = True
        plt.rcParams['figure.figsize'] = [15.0, 7.0]
        plt.rcParams['savefig.dpi'] = 96
        plt.rcParams['font.size'] = 12
        plt.rcParams['legend.fontsize'] = 'medium'
        plt.rcParams['figure.titlesize'] = 'medium'

    def price_view(self, df, b_date, s_cd, size=(15, 7), make_file=False):

        cds = common.str_list(s_cd)

        fig, ax = plt.subplots(figsize=size)
        plt.show()
        x = df.loc[b_date:].index

        plt.autoscale(True, axis='both')

        for c in cds:
            plt.plot(x, df.loc[b_date:, c], label=c)

        x_length = len(x)
        jump = int(x_length / 10)
        xs = list()
        for i in range(10):
            xs.append(x[jump * i])
        xs.append(x[-1])
        plt.xticks(np.arange(0, x_length + jump, jump), xs, rotation=45)

        plt.legend()
        plt.show()
        if make_file:
            plt.savefig('./image/' + VisualizeIntraday.today + cds[0] + ' price_view.png', bbox_inches='tight')

    def index_view(self, df, b_date, s_cd, size=(10, 7), make_file=False):

        fig, ax = plt.subplots(1, 1, figsize=size)
        x = df.loc[b_date:].index

        plt.autoscale(True, axis='both')

        cds = common.str_list(s_cd)

        for c in cds:
            plt.plot(x, df.loc[b_date:, c] / df.loc[b_date, c] * 100, label=c)

        x_length = len(x)
        jump = int(x_length / 10)
        xs = list()
        for i in range(10):
            xs.append(x[jump * i])
        xs.append(x[-1])
        plt.xticks(np.arange(0, x_length + jump, jump), xs, rotation=45)

        plt.legend()

        if make_file:
            plt.savefig('./image/' + Visualize.today + s_cd[0] + ' index_view.png', bbox_inches='tight')

    def complex_view(self, df, b_date, cd_set_a, cd_set_b=[], size=(15, 7), make_file=False):

        cds_a = common.str_list(cd_set_a)
        cds_b = common.str_list(cd_set_b)

        fig, ax1 = plt.subplots(figsize=size)
        x = df.loc[b_date:].index

        plt.autoscale(True, axis='both')

        i = 1
        for c in cds_a:
            if i == 1:
                ax1.plot(x, df.loc[b_date:, c], color='C' + str(i), lw=3, label=c)
            else:
                ax1.plot(x, df.loc[b_date:, c], color='C' + str(i), label=c)
            i += 1

        if cds_b:
            ax2 = ax1.twinx()

            i = 6
            for c in cds_b:
                ax2.fill_between(x, df.loc[b_date:, c], 0, facecolor='C' + str(i), alpha=0.3)
                ax1.plot(np.nan, color='C' + str(i), label=c)
                i += 1

        x_length = len(x)
        jump = int(x_length / 10)
        xs = list()
        for i in range(10):
            xs.append(x[jump * i])
        xs.append(x[-1])

        ax1.set_xticks(np.arange(0, x_length + jump, jump))
        ax1.set_xticklabels(xs, rotation=45)
        ax2.set_xticks(np.arange(0, x_length + jump, jump))
        ax2.set_xticklabels(xs, rotation=45)

        ax1.legend(loc=0)

        if make_file:
            plt.savefig('./image/' + Visualize.today + cds_a[0] + ' complex_view.png', bbox_inches='tight')

    def multi_line_view(self, df, b_date, cd_set_a, cd_set_b=[], size=(15, 7), make_file=False):

        cds_a = common.str_list(cd_set_a)
        cds_b = common.str_list(cd_set_b)

        fig, ax1 = plt.subplots(figsize=size)
        x = df.loc[b_date:].index

        plt.autoscale(True, axis='both')

        i = 1
        for c in cds_a:
            if i == 1:
                ax1.plot(x, df.loc[b_date:, c], color='C' + str(i), lw=3, label=c)
                pass
            else:
                ax1.plot(x, df.loc[b_date:, c], color='C' + str(i), label=c)
            i += 1

        if cds_b:
            ax2 = ax1.twinx()

            i = 6
            for c in cds_b:
                ax2.plot(x, df.loc[b_date:, c], color='C' + str(i), label=c, alpha=0.7)
                ax1.plot(np.nan, color='C' + str(i), label=c)
                i += 1

        x_length = len(x)
        jump = int(x_length / 10)
        xs = list()
        for i in range(10):
            xs.append(x[jump * i])
        xs.append(x[-1])

        ax1.set_xticks(np.arange(0, x_length + jump, jump))
        ax1.set_xticklabels(xs, rotation=45)
        ax2.set_xticks(np.arange(0, x_length + jump, jump))
        ax2.set_xticklabels(xs, rotation=45)

        ax1.legend(loc=0)

        if make_file:
            plt.savefig('./image/' + Visualize.today + cds_a[0] + ' multi_line_view.png', bbox_inches='tight')

    def position_view(self, df, s_cd, size=(15, 1), make_file=False):
        cds = common.str_list(s_cd)

        fig, ax = plt.subplots(figsize=size)
        x = df.index

        for c in cds:
            df['ps' + c] = 0
            df.loc[df['p ' + c] == 'll', ['ps' + c]] = 1
            df.loc[df['p ' + c] == 'sl', ['ps' + c]] = 1
            df.loc[df['p ' + c] == 'zl', ['ps' + c]] = 1
            df.loc[df['p ' + c] == 'ls', ['ps' + c]] = -1
            df.loc[df['p ' + c] == 'ss', ['ps' + c]] = -1
            df.loc[df['p ' + c] == 'zs', ['ps' + c]] = -1
            plt.fill_between(x, df['ps' + c], 0, label=c)
        plt.yticks([-1, 0, 1], ["Short", "Zero", "Long"])

        x_length = len(x)
        jump = int(x_length / 10)
        xs = list()
        for i in range(10):
            xs.append(x[jump * i])
        xs.append(x[-1])
        plt.xticks(np.arange(0, x_length + jump, jump), xs, rotation=45)

        plt.legend()

        if make_file:
            plt.savefig('./image/' + VisualizeIntraday.today + cds[0] + ' position_view.png', bbox_inches='tight')

    def pair_trend_price_view(self, df, thd, s_cd, make_file=False, size=(15, 7)):
        fig, ax = plt.subplots(figsize=size)
        x = df.index

        plt.fill_between(x, df[s_cd[1] + ' expected'] * (1 + thd), df[s_cd[1] + ' expected'] * (1 - thd),
                         facecolor='sienna', alpha=0.2)
        plt.plot(x, df[s_cd[1] + ' expected'], 'sienna', linestyle='--')
        plt.plot(x, df[s_cd[0]], 'C0')
        plt.plot(x, df[s_cd[1]], 'C1', lw=3)
        plt.legend()
        if make_file:
            plt.savefig('./image/' + VisualizeIntraday.today + s_cd[0] + ' pairs_trend_price_view.png',
                        bbox_inches='tight')

    def pair_trend_index_view(self, df, thd, s_cd, make_file=False, size=(15, 7)):
        fig, ax1 = plt.subplots(figsize=size)
        x = df.index

        ax1.fill_between(x, df[s_cd[1] + ' expected'] * (1 + thd), df[s_cd[1] + ' expected'] * (1 - thd),
                         facecolor='sienna', alpha=0.2)
        ax1.plot(x, df[s_cd[1] + ' expected'], 'sienna', linestyle='--')
        ax1.plot(x, df[s_cd[1]], 'C1', lw=3)

        ax2 = ax1.twinx()
        ax2.plot(x, df[s_cd[0]], 'C0', alpha=0.7)
        ax1.plot(np.nan, 'C0', label=s_cd[0])

        x_length = len(x)
        jump = int(x_length / 10)
        xs = list()
        for i in range(10):
            xs.append(x[jump * i])
        xs.append(x[-1])

        ax1.set_xticks(np.arange(0, x_length + jump, jump))
        ax1.set_xticklabels(xs, rotation=45)
        ax2.set_xticks(np.arange(0, x_length + jump, jump))
        ax2.set_xticklabels(xs, rotation=45)

        ax1.legend(loc=0)
        if make_file:
            plt.savefig('./image/' + VisualizeIntraday.today + s_cd[0] + ' pairs_trend_index_view.png',
                        bbox_inches='tight')

    def BB_trend_view(self, sample, sigma, s_cd, make_file=False, size=(15, 7)):
        cds = common.str_list(s_cd)

        fig, ax = plt.subplots(figsize=size)
        x = sample.index

        plt.fill_between(x, sample['lb'], sample['ub'], facecolor='sienna', alpha=0.2)
        plt.plot(x, sample['center'], color='sienna', linestyle='--', label='MA')
        plt.plot(x, sample[cds[0]], color='C0', linestyle='-', lw=3)

        x_length = len(x)
        jump = int(x_length / 10)
        xs = list()
        for i in range(10):
            xs.append(x[jump * i])
        xs.append(x[-1])
        plt.xticks(np.arange(0, x_length + jump, jump), xs, rotation=45)

        plt.legend()
        if make_file:
            plt.savefig('./image/' + VisualizeIntraday.today + cds[0] + ' bb_trend_view.png', bbox_inches='tight')

    def futures_basis_view(self, df, threshold, s_cd, make_file=False, size=(15, 7)):

        cds = common.str_list(s_cd)

        fig, ax = plt.subplots(figsize=size)
        x = df.index
        plt.fill_between(x, df[cds[0]], df[cds[0]] + df['basis'], facecolor='sienna', alpha=0.2)
        plt.plot(x, df[cds[0]], 'sienna', linestyle='--')
        plt.plot(x, df[cds[1]], 'C1', lw=3)

        x_length = len(x)
        jump = int(x_length / 10)
        xs = list()
        for i in range(10):
            xs.append(x[jump * i])
        xs.append(x[-1])
        plt.xticks(np.arange(0, x_length + jump, jump), xs, rotation=45)

        plt.legend()
        if make_file:
            plt.savefig('./image/' + VisualizeIntraday.today + cds[0] + ' futures_basis_view.png', bbox_inches='tight')


class Visualize3D():
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    def __init__(self):
        plt.style.use('fivethirtyeight')
        plt.rcParams['font.family'] = font
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['axes.grid'] = True
        plt.rcParams['lines.linewidth'] = 1.5
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.alpha'] = 0.7
        plt.rcParams['lines.antialiased'] = True
        plt.rcParams['figure.figsize'] = [15.0, 7.0]
        plt.rcParams['savefig.dpi'] = 96
        plt.rcParams['font.size'] = 12
        plt.rcParams['legend.fontsize'] = 'medium'
        plt.rcParams['figure.titlesize'] = 'medium'

    def surface_view(self, size=(10, 6), **points):
        labels = []
        values = []

        for key, value in points.items():
            labels.append(key)
            values.append(value)

        try:
            fig = plt.figure(figsize=size)
            ax = fig.gca(projection='3d')

            surf = ax.plot_surface(values[0], values[1], values[2], cmap=self.cm.summer, linewidth=1, alpha=0.8)

            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_zlabel(labels[2])

            fig.colorbar(surf, shrink=0.5, aspect=5)

        except Exception as e:
            print('x, y, z 각 축 입력값의 개수가 일치해야 합니다.')
            print(e)