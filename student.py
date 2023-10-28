import pandas as pd

dataFile = pd.read_csv("./linreg_datasets_WS2021 (1)/random3.csv",header=None)
y_train = list(dataFile[2])
x1 = list(dataFile[0])
x2 = list(dataFile[1])
threshold = 0.0001
eta = 0.00005
j = 0
w0 = 0
w1 = 0
w2 = 0
y = 0
prev_sse = 0
while 1:
    sse = 0
    g0 = 0
    g1 = 0
    g2 = 0
    for i in range(0,len(y_train)):
        y = w0 + (w1*x1[i]) + (w2*x2[i])
        g0 = g0 + (y_train[i] - y)
        g1 = g1 + (x1[i]*(y_train[i] - y))
        g2 = g2 + (x2[i]*(y_train[i] - y))
        sse = sse + (y - y_train[i])**2
    print('{0},{1},{2},{3},{4}'.format(j, w0, w1, w2, sse), end='\n')
    if(j>0):
        if (prev_sse - sse) < threshold:
            break
    # Updating weights
    w0 = w0 + eta*g0
    w1 = w1 + eta*g1
    w2 = w2 + eta*g2
    j += 1
    prev_sse = sse





