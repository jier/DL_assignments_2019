import sys
sys.path.append('..')
import numpy as np 
import csv 
import matplotlib.pyplot as plt 

#ENCODING OF THE DATA IS ALL STRINGS SO MANUALLY EDITING DATA TO FLOAT IN ORDER TO PLOT WELL
#INSPECT THE .csv files last row entries will correspond with below values

def plot_from_csv_lstm():
    mean = [ 1,  1,  1, 1,  0.27500001,  0.1140625,
    0.08125 ,0.1, 0.1 , 0.0953125,   0.1015625 ]
    std = [ 0,       0,0 ,0,0.36290389, 0.03617449,
        0.01887976,  0.01939012,  0.02061079,  0.00911086,  0.01307281]
    
    # with open(filename, 'r', encoding='utf-8') as csvfile:
    #     data = csv.reader(csvfile, delimiter=",")
    #     tempx = []
    #     tempy = []
    #     ticks = []
    #     for item in data:
            
    #         tempx.append(item[1].strip('').replace('\n','').strip())
    #         tempy.append(item[2].strip('').replace('\n','').strip()) 
    #         ticks.append(item[3].strip('').replace('\n','').strip())
    #     mean, std, steps = np.array(tempx)[-1], np.array(tempy)[-1] , np.array(ticks)[1:]
        
        # print(list(map(float, mean)))
    
    fig, ax = plt.subplots(figsize=(15, 5))
    x = range(5, 60, 5)
    lower = [ mean[i] - std[i] for i in range(len(mean))]
    upper = [ mean[i] + std[i] for i in range(len(mean))]
    ax.fill_between(x, lower, upper, facecolor='blue', alpha=0.5 )
    ax.plot(x, mean, color='blue')
    ax.set_xlabel("T")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Palindrome Test over time LSTM")
    ax.grid()
    plt.savefig('LSTM_acc_.pdf', format='pdf')



def plot_from_csv_rnn():

    mean = [ 1, 1, 1 , 1 ,1, 1,1,
    1,         1,          0.7453125,  0.81718749,  0.54843748,  0.65937501,
    0.45468751,  0.49062499,  0.27031249,  0.58281249,  0.203125,    0.53593749,
    0.21250001,  0.153125,    0.16249999,  0.3125,      0.28437501,  0.21875   ]
    std =[ 0, 0 ,0,  0 ,          0,  0,  0, 
    0,          0,          0.35546786,  0.36562499,  0.38249031,  0.41790959,
    0.34482786,  0.36731714,  0.15353897,  0.36894858,  0.09164299,  0.3849462,
    0.03610694,  0.06222596,  0.07818747,  0.21383999,  0.10741548,  0.08370685]

    fig, ax = plt.subplots(figsize=(15, 5))
    x = range(0, len(mean))
    lower = [ mean[i] - std[i] for i in range(len(mean))]
    upper = [ mean[i] + std[i] for i in range(len(mean))]
    ax.fill_between(x, lower, upper, facecolor='blue', alpha=0.5 )
    ax.plot(x, mean, color='blue')
    ax.set_xlabel("T")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Palindrome Test over time RNN")
    ax.grid()
    plt.savefig('RNN_acc_.pdf', format='pdf')

if __name__ == "__main__":

    # plot_from_csv_lstm()
    plot_from_csv_rnn()
