# import librarys
import json
import time
import datetime
import calendar
import krakenex

# query crypto data from kraken api
def get_crypto_data(pair, since):
    return api.query_public('OHLC', data={'pair':pair, 'since':since})['result'][pair]



# print a more readable list of candles
def print_candles(list_of_candles):

    #loop through the list of candles
    for candle in list_of_candles:
        # setting variables for the open and close price as well as the highest and lowest price of the current candle
        open_ = float(candle[1])
        high_ = float(candle[2])
        low_ = float(candle[3])
        close_ = float(candle[4])
        print('O: ' + str(open_) + '\nH: ' + str(high_) + '\nL: ' + str(low_) + '\nC: ' + str(close_))
        print('\n')



# analyze crypto pair for tradehistory within a timespan and buy or sell depending on history
def analyze(pair, since):
    # get crypto data from kraken api by passing the crypto pair tuple as concatenated argument as well as timespan
    data = get_crypto_data(pair[0]+pair[1], since)
    # print_candles(data)

    # set variables for the lowest and highest prices within defined timespan to 0
    lowest = 0
    highest = 0

    # loop through prices of every minute of defined timespan
    for prices in data:
        # setting variables for the current balance, the last trade (buy) and the price of the last trade (buy)
        balance = get_fake_balance()
        last_trade = get_last_trade(pair[0]+pair[1])
        last_trade_price = float(last_trade['price'])



        # setting variables for the open and close price as well as the highest and lowest price of the current minute
        open_ = float(prices[1])
        high_ = float(prices[2])
        low_ = float(prices[3])
        close_ = float(prices[4])


        # setting a bool variable if a crypto currency has already been sold
        did_sell = False

        try:
            balance[pair[0]]
            # print(balance[pair[0]])
            #if we own any of the pair currency that we are looking at then check sell
            selling_point_win = last_trade_price * 1.005
            selling_point_loss =  last_trade_price * 0.995

            # selling at a win
            if open_ >= selling_point_win or close_ >= selling_point_win:
                # sell at a profit
                did_sell = True
                fake_sell(pair, close_, last_trade)
                # print current balance for each iteration, last_trade (= Buy) and last trades (= Buy) price
                print('Balance: ' + json.dumps(balance, indent=4) + '\n\nLast Trade: ' + json.dumps(last_trade, indent = 4) + '\n\nLast Trade Price: ' + json.dumps(last_trade_price, indent=4))
                print('Current candle:\nO: ' + str(open_) + '\nH: ' + str(high_) + '\nL: ' + str(low_) + '\nC: ' + str(close_))
                print('\n')
                print('sold ' + str(balance[pair[0]]) + ' ETH (take profit) at: ' + str(close_))
                print('new balance: ' + str(get_fake_balance()))
                print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
            elif open_ <= selling_point_loss or close_ <= selling_point_loss:
                # sell at a loss
                did_sell = True
                fake_sell(pair, close_, last_trade)
                print('Balance: ' + json.dumps(balance, indent=4) + '\n\nLast Trade: ' + json.dumps(last_trade, indent = 4) + '\n\nLast Trade Price: ' + json.dumps(last_trade_price, indent=4))
                print('Current candle:\nO: ' + str(open_) + '\nH: ' + str(high_) + '\nL: ' + str(low_) + '\nC: ' + str(close_))
                print('\n')
                print('sold ' + str(balance[pair[0]]) + ' ETH (stop loss) at: ' + str(close_))
                print('new balance: ' + str(get_fake_balance()))
                print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')

        except:
            pass

        # logic for if we should buy
        if not did_sell and float(balance['USD.HOLD']) > 0:
            if low_ < lowest or lowest == 0:
                lowest = low_
            if high_ > highest:
                highest = high_

            price_to_buy = 1.0005

            if highest/lowest >= price_to_buy and low_ <= lowest:
                available_money = balance['USD.HOLD']
                #buy
                fake_buy(pair, available_money, close_, last_trade)
                # print('bought ETH at: '+ str(close_))
                print('Balance: ' + json.dumps(balance, indent=4) + '\n\nLast Trade: ' + json.dumps(last_trade, indent = 4) + '\n\nLast Trade Price: ' + json.dumps(last_trade_price, indent=4))
                print('Current candle:\nO: ' + str(open_) + '\nH: ' + str(high_) + '\nL: ' + str(low_) + '\nC: ' + str(close_))
                print('\n')
                print('bought ' + get_fake_balance()[pair[0]] + ' ETH at: '+ str(close_))
                print('new balance: ' + str(get_fake_balance()))
                # print(get_fake_balance()[pair[0]])

                print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')



def fake_update_balance(pair, dollar_amount, close_, was_sold):
    balance = get_fake_balance()
    prev_balance = float(balance['USD.HOLD'])
    new_balance = 0
    if was_sold:
        new_balance = prev_balance + float(dollar_amount)
        del balance[pair[0]]
    else:
        new_balance = prev_balance - float(dollar_amount)
        balance[pair[0]] = str(float(dollar_amount)/close_)
    balance['USD.HOLD'] = str(new_balance)

    with open('balance.json', 'w') as f:
        json.dump(balance, f, indent = 4)


def fake_buy(pair, dollar_amount, close_, last_trade):
    # buying with the real account: api.query_private('order', data, ...)
    # figure out by myself. Is not covered in the tutorial
    trades_history = get_fake_trades_history()
    last_trade['price'] = str(close_)
    last_trade['type'] = 'buy'
    last_trade['cost'] = dollar_amount
    last_trade['time'] = datetime.datetime.now().timestamp()
    last_trade['vol'] = str(float(dollar_amount)/close_)

    trades_history['result']['trades'][str(datetime.datetime.now().timestamp())] = last_trade
    with open('tradeshistory.json', 'w') as f:
        json.dump(trades_history, f, indent = 4)
        fake_update_balance(pair, dollar_amount, close_, False)


def fake_sell(pair, close_, last_trade):
    trades_history = get_fake_trades_history()
    last_trade['price'] = str(close_)
    last_trade['type'] = 'sell'
    last_trade['cost'] = str(float(last_trade['vol'])*close_)
    last_trade['time'] = datetime.datetime.now().timestamp()

    trades_history['result']['trades'][str(datetime.datetime.now().timestamp())] = last_trade

    with open('tradeshistory.json', 'w') as f:
        json.dump(trades_history, f, indent = 4)
        fake_update_balance(pair, float(last_trade['cost']), close_, True)


def get_balance():
    return api.query_private('Balance')


def get_fake_balance():
    with open('balance.json', 'r') as f:
        return json.load(f)


def get_last_trade(pair):
    trades_history = get_fake_trades_history()['result']['trades']

    last_trade = {}

    for trade in trades_history:
        trade = trades_history[trade]
        if trade['pair'] == pair and trade['type'] == 'buy':
            last_trade = trade
    return last_trade


def get_trades_history():
    start_date = datetime.datetime(2020, 1, 1)
    end_date = datetime.datetime.today()
    return api.query_private('TradesHistory', req(start_date, end_date, 1))#['result']['trades']


def get_fake_trades_history():
    with open('tradeshistory.json', 'r') as f:
        return json.load(f)


def date_nix(str_date):
    return calendar.timegm(str_date.timetuple())

def req(start, end, ofs):
    req_data = {
        'type': 'all',
        'trades': 'true',
        'start': str(date_nix(start)),
        'end': str(date_nix(end)),
        'ofs': str(ofs)
    }
    return req_data


if __name__ == '__main__':
    api = krakenex.API()
    api.load_key('../../kraken.key')
    pair = ("XETH", "ZUSD")
    since = str(int(time.time()-3600))
    #print(json.dumps((get_crypto_data(pair[0]+pair[1], since)), indent=4))
    analyze(pair, since)
