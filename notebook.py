#NOMURA QUANT CHALLENGE 2025

#The format for the weights dataframe for the backtester is attached with the question.
#Complete the below codes wherever applicable

import pandas as pd


def backtester_without_TC(weights_df):
    #Update data file path here
    data = pd.read_csv('file_path')

    weights_df = weights_df.fillna(0)

    start_date = 3500
    end_date = 3999

    initial_notional = 1

    df_returns = pd.DataFrame()

    for i in range(0,20):
        data_symbol = data[data['Symbol']==i]
        data_symbol = data_symbol['Close']
        data_symbol = data_symbol.reset_index(drop=True)   
        data_symbol = data_symbol/data_symbol.shift(1) - 1
        df_returns =  pd.concat([df_returns,data_symbol], axis=1, ignore_index=True)
    
    df_returns = df_returns.fillna(0)
    
    weights_df = weights_df.loc[start_date:end_date]    
    df_returns = df_returns.loc[start_date:end_date]

    df_returns = weights_df.mul(df_returns)

    notional = initial_notional

    returns = []

    for date in range(start_date,end_date+1):
        returns.append(df_returns.loc[date].values.sum())
        notional = notional * (1+returns[date-start_date])

    net_return = ((notional - initial_notional)/initial_notional)*100
    sharpe_ratio = (pd.DataFrame(returns).mean().values[0])/pd.DataFrame(returns).std().values[0]

    return [net_return, sharpe_ratio]



def task1_Strategy1():
    train_data = pd.read_csv('train_data.csv')
    crossval_data = pd.read_csv('crossval_data.csv')
    output_df = pd.DataFrame()  #output_df is the output dataframe containing weights

    #Write your code here
    
    return output_df


def task1_Strategy2():
    train_data = pd.read_csv('train_data.csv')
    crossval_data = pd.read_csv('crossval_data.csv')
    output_df = pd.DataFrame()  #output_df is the output dataframe containing weights

    #Write your code here
    
    return output_df


def task1_Strategy3():
    train_data = pd.read_csv('train_data.csv')
    crossval_data = pd.read_csv('crossval_data.csv')
    output_df = pd.DataFrame()  #output_df is the output dataframe containing weights

    #Write your code here
    
    return output_df


def task1_Strategy4():
    train_data = pd.read_csv('train_data.csv')
    crossval_data = pd.read_csv('crossval_data.csv')
    output_df = pd.DataFrame()  #output_df is the output dataframe containing weights

    #Write your code here
    
    return output_df


def task1_Strategy5():
    train_data = pd.read_csv('train_data.csv')
    crossval_data = pd.read_csv('crossval_data.csv')
    output_df = pd.DataFrame()  #output_df is the output dataframe containing weights

    #Write your code here
    
    return output_df


def task1():
    Strategy1 = task1_Strategy1()
    Strategy2 = task1_Strategy2()
    Strategy3 = task1_Strategy3()
    Strategy4 = task1_Strategy4()
    Strategy5 = task1_Strategy5()

    performanceStrategy1 = backtester_without_TC(Strategy1)
    performanceStrategy2 = backtester_without_TC(Strategy2)
    performanceStrategy3 = backtester_without_TC(Strategy3)
    performanceStrategy4 = backtester_without_TC(Strategy4)
    performanceStrategy5 = backtester_without_TC(Strategy5)

    output_df = pd.DataFrame({'Strategy1':performanceStrategy1, 'Strategy2': performanceStrategy2, 'Strategy3': performanceStrategy3, 'Strategy4': performanceStrategy4, 'Strategy5': performanceStrategy5})
    output_df.to_csv('task1.csv')
    return



def task2():
    output_df_weights = pd.DataFrame()
    
    #Write your code here

    output_df_weights.to_csv('task2_weights.csv')
    results = backtester_without_TC(output_df_weights)
    df_performance = pd.DataFrame({'Net Returns': [results[0]], 'Sharpe Ratio': [results[1]]})
    df_performance.to_csv('task_2.csv')
    return



def calculate_turnover(weights_df):
    weights_diff_df = abs(weights_df-weights_df.shift(1))
    turnover_symbols = weights_diff_df.sum()
    turnover = turnover_symbols.sum()
    return turnover

def backtester_with_TC(weights_df):
    #Update path for data here
    data = pd.read_csv('file_path')

    weights_df = weights_df.fillna(0)

    turnover = calculate_turnover(weights_df)

    start_date = 3000
    end_date = 3499

    transaction_cost = (turnover * 0.01)

    df_returns = pd.DataFrame()

    for i in range(0,20):
        data_symbol = data[data['Symbol']==i]
        data_symbol = data_symbol['Close']
        data_symbol = data_symbol.reset_index(drop=True)   
        data_symbol = data_symbol/data_symbol.shift(1) - 1
        df_returns =  pd.concat([df_returns,data_symbol], axis=1, ignore_index=True)
    
    df_returns = df_returns.fillna(0)
    
    weights_df = weights_df.loc[start_date:end_date]    
    df_returns = df_returns.loc[start_date:end_date]

    df_returns = weights_df.mul(df_returns)

    initial_notional = 1
    notional = initial_notional

    returns = []

    for date in range(start_date,end_date+1):
        returns.append(df_returns.loc[date].values.sum())
        notional = notional * (1+returns[date-start_date])

    net_return = ((notional - transaction_cost - initial_notional)/initial_notional)*100
    sharpe_ratio = (pd.DataFrame(returns).mean().values[0] - (transaction_cost/(end_date-start_date+1)))/pd.DataFrame(returns).std().values[0]

    return [net_return, sharpe_ratio]



def task3():
    output_df_weights = pd.DataFrame()
    
    #Write your code here

    output_df_weights.to_csv('task3_weights.csv')
    results = backtester_with_TC(output_df_weights)
    df_performance = pd.DataFrame({'Net Returns': [results[0]], 'Sharpe Ratio': [results[1]]})
    df_performance.to_csv('task_3.csv')
    return



if __name__ == '__main__':
    task1()
    task2()
    task3()