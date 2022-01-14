# import librarys
import json #use json operations
import numpy as np #use numpy calculation operations
import pandas as pd #use pandas dataframe operations
import time #use time methods
import datetime #use date methods
import calendar #use calendar methods
import krakenex #access the kraken API
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score #get model performance metrics
from sklearn.model_selection import train_test_split #create random train/test split
from sklearn.svm import SVC # ML Model
#from scitime import RuntimeEstimator



def moving_average(series, N):
    return series.rolling(N).mean()


def assign_labels(dataframe):
# if the dataframe['return'] > 0, label as 1 otherwise as 0
    dataframe['label'] = np.where(dataframe['return'] > 0, 1, 0)

    return dataframe


def get_OHLC_dataframe(csv):
# read data from Kraken provided OHLC csv files and return pandas dataframe
    OHLC_df = pd.read_csv(csv, header = None, names= ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trades'])
    return OHLC_df



def prepare_dataset(candle_interval, num_candles):
# parameters:
# candle_interval - integer that defines the candlestick time interval that is analyzed (choose between 1, 5, 15, 60, 720, 1440 minutes) - each number accesses a different csv file in the directory
# num_candles - integer that defines the number of candles (sample size) that are used for training and testing the model. Always the newest candles are chosen. Higher numbers lead to exponentially higher modelling times.


    # accesses the appropriate csv file corresponding to defined candle_interval
    csv = 'ETHUSDT_{}.csv'.format(candle_interval)

    # read data from Kraken provided OHLC csv file and save it into pandas dataframe
    dataframe = get_OHLC_dataframe(csv)


    # drop all candles except for the last X (num_candles) - sample size
    df_rows = dataframe.shape[0]
    dataframe = dataframe.iloc[df_rows-num_candles:]


    # set datetime datatype as dataframe index for being able to use the pandas shift method later
    # drop the Timestamp column
    dataframe = dataframe.set_index(pd.to_datetime(dataframe['Timestamp'], unit='s'))
    dataframe = dataframe.drop(columns=['Timestamp'])

    # adding the (20, 50, 200) moving averages as well as their differences to the candles closing values to the dataframe
    dataframe['moving_average_20'] = moving_average(dataframe.Close, 20)
    dataframe['diff_mov_avg_20'] = dataframe.Close / dataframe.moving_average_20

    dataframe['moving_average_50'] = moving_average(dataframe.Close, 50)
    dataframe['diff_mov_avg_50'] = dataframe.Close / dataframe.moving_average_50

    dataframe['moving_average_200'] = moving_average(dataframe.Close, 200)
    dataframe['diff_mov_avg_200'] = dataframe.Close / dataframe.moving_average_200


    # add column with return ratio to dataframe by substracting the closing price of the
    # preceding timestamp from the current timestamp and dividing it by the closing price of the
    # preceding timestamp
    dataframe['return'] = (dataframe.Close - dataframe.Close.shift(1))/dataframe.Close.shift(1)

    # adding the log return (ln(r+1)) as a new column to the dataframe
    dataframe['log_return'] = np.log(dataframe['return']+1)

    # assigns 0 or 1 to each candle if its return is above the threshold of 0
    # if it is classified with 1 it is a candle that we want to buy when it opens
    dataframe = assign_labels(dataframe)


    return dataframe





def train_model(dataframe, num_feat_candles, gamma, C):
# parameters:
# num_feat_candles - integer that defines the number of preceding candles that are considered to predict the candle of interest
# gamma and C - factors that describe the shape of the corresponding higher dimensional 'curve' that separate the datapoints within our dataset from each other


# We want to take the preceding X candles (num_feat_candles) into consideration for predicting the label of the candle of interest
# We store the preceding X candle 'returns' as predicting features in an array of arrays in the variable np_features
# We store the corresponding labels for the candle of interest in the variable np_labels


    for z in range(num_feat_candles):
    # add X (num_feat_candles) columns to the dataframe, each new column showing the return value of the 1 to X preceding candles return values to predict the label of the next candle
        dataframe['pos_{}'.format(z+1)] = dataframe['return'].shift(z+1)

    #gets the 20 moving average from the preceding candle as feature to predict the label of the next candle
    dataframe['pos_mvg_avg_20'] = dataframe['diff_mov_avg_20'].shift(1)



    # create numpy arrays for features and labels from df_class
    # feature values are calculated from the X (num_feat_candles) preceding candles

    df_np_features = dataframe.iloc[: , -num_feat_candles-1:]
    df_np_label = dataframe['label']

    '''
    #Print for control:
    df_np_features['diff_mov_avg_20'] = dataframe['diff_mov_avg_20']
    df_np_features['return'] = dataframe['return']

    print(df_np_features.shape)
    print(df_np_label.shape)
    print(df_np_features.head(25))
    print(df_np_label.head(25))
    exit()
    '''

    # the first X (num_feat_candles+1) return values in the list cannot be calculated, since there are not enough preceding values to calculate a feature from
    # the first 20 candles have a NaN value for the moving average, so we drop 20 candles and have also dropped the candles without return values
    np_features = df_np_features.iloc[20: ].to_numpy()
    np_label = df_np_label.iloc[20:].to_numpy()


    '''
    print(np_features)
    print(np_features.shape)
    print(np_label)
    print(np_label.shape)
    exit()
    '''


    # split the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(np_features, np_label, test_size = 0.2, random_state=42)

    # print('Number of rows in the training set: {}'.format(X_train.shape[0]))
    # print('Number of rows in the test set: {}'.format(X_test.shape[0]))

    # initialize a model
    model = SVC(kernel='rbf', gamma = gamma, C=C)

    # calculate predicted modeling time
    #estimator = RuntimeEstimator(meta_algo='RF', verbose = 3)
    #estimation, lower_bound, upper_bound = estimator.time(model, X_train, y_train)
    #print(estimation, lower_bound, upper_bound)

    # fitting the model
    model.fit(X_train, y_train)

    # test the model by giving it testing data to predict the labels for
    test_pred = model.predict(X_test)

    #print(y_test)
    #print(test_pred)

    # calculate scores by comparing the predicted labels with the actual labels from the testset in order to show the accuracy of the model

    # The f1 score is an overall score that measures the accuracy of a test and is calculated from the precision and recall of that test
    # The precision is the number of true positive results divided by the number of all positive results, including those not identified correctly
    # The recall is the number of true positive results divided by the number of all samples that should have been identified as positive.
    _acc_score = accuracy_score(y_test, test_pred)
    _prec_score = precision_score(y_test, test_pred)
    _rec_score = recall_score(y_test, test_pred)
    _f1 = f1_score(y_test, test_pred)

    return _acc_score, _prec_score, _rec_score, _f1



def run_parameter_training (dataframe, num_feat_candles_max, gamma_max, C_max):
# trains multiple models with the parameters num_feat_candles_max, gamma and C in the range from 0 to the input parameters given to the function

    # initialize variables for the optimal parameters as well as for the loop number
    loop_number = 0
    accuracy_score_best = 0.0
    precision_score_best = 0.0
    recall_score_best = 0.0
    f1_score_best = 0.0

    for i in range(1, num_feat_candles_max+1):
        for j in range(1, gamma_max+1):
            for k in range(1, C_max+1):

                # train the model with the variable input parameters num_feat_candles_max, gamma and C
                acc_score, prec_score, rec_score, f1 = train_model(dataframe, num_feat_candles = i, gamma = j, C = k)


                if (f1 > f1_score_best):
                # If the current model training iteration with the corresponding parameters reveals an f1 score that is
                # higher than the f1 scores so far, save the corresponding accuracy, precision, recall and f1 score to
                # 'best' variables as well as the corresponding parameters for later replicability and print everything.

                    '''
                    # alternative condition for defining best scores
                    if ((acc_score >= accuracy_score_best)
                        and (prec_score >= precision_score_best)
                        and (rec_score >= recall_score_best)
                        and (f1 >= f1_score_best)):
                    '''

                    # save the scores
                    accuracy_score_best = acc_score
                    precision_score_best = prec_score
                    recall_score_best = rec_score
                    f1_score_best = f1

                    # save the parameters
                    best_num_feat_candles = i
                    best_gamma = j
                    best_C = k

                    # print scores and corresponding parameters
                    print('\nBest values so far: ')
                    print('Accuracy Score: ' + str(accuracy_score_best))
                    print('Precision Score: ' + str(precision_score_best))
                    print('Recall Score: ' + str(recall_score_best))
                    print('F1 Score: ' + str(f1_score_best))
                    print(
                    'with number of featured candles = ' + str(best_num_feat_candles) +
                    ', gamma = ' + str(best_gamma) +
                    ' and C = ' + str(best_C))

                # count the loop number
                loop_number += 1

                if(loop_number % 100 == 0):
                # every 100 loops print the loop number and the current parameters the model is checking
                    print('\nLoop Number: ' + str(loop_number))
                    print('Current parameters checking: Featured Candles = ' + str(i) + ', gamma = ' + str(j) + ' and C = ' + str(k))

    # after all models with the parameters in the defined intervals have been trained, print the scores of the model and the corresponding
    # parameters with the best scores.
    print('\n\nFinal Scores')
    print('Accuracy Score: ' + str(accuracy_score_best))
    print('Precision Score: ' + str(precision_score_best))
    print('Recall Score: ' + str(recall_score_best))
    print('F1 Score: ' + str(f1_score_best))
    print('With number of featured candles = ' + str(best_num_feat_candles) + ', gamma = ' + str(best_gamma) + ' and C = ' + str(best_C))




if __name__ == '__main__':
    api = krakenex.API()
    api.load_key('../../kraken.key')
    pair = ("XETH", "ZUSD")

    # prepare the dataset
    # candle_interval - integer that defines the candlestick time interval that is analyzed (choose between 1, 5, 15, 60, 720, 1440 minutes) - each number accesses a different csv file in the directory
    # num_candles - integer that defines the number of candles (sample size) that are used for training and testing the model. Always the newest candles are chosen. Higher numbers lead to exponentially(?) higher modelling times.
    df = prepare_dataset(candle_interval = 60, num_candles = 1000)

    # train a model on the given dataset with concrete parameters and print the obtained scores
    #accuracy_score, precision_score, recall_score, f1_score = train_model(df, num_feat_candles = 4, gamma = 18, C = 2)
    #print(accuracy_score, precision_score, recall_score, f1_score)

    # train multiple models with the parameters num_feat_candles_max, gamma and C in the range from 0 to the input parameters given to the function
    # increase the parameters gradually as it results in higher processing duration
    run_parameter_training(df, 4, 20, 20)
