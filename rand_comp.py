

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time



#データ読み込み
data = pd.read_csv('ml-100k/ml-100k/u.data', sep="\t", header= None)

#評価行列の生成
npdata = np.zeros((944, 1683))
npdata_random = np.zeros((944, 1683))
for i in range(len(data)):
    npdata[int(data[0][i])][data[1][i]] = data[2][i]

###user 乱数による初期化
for i in range(943):
    for j in  range(1683):
        npdata_random[i][j] = random.randint(1,5)

#npdata の　一行目　一列目　其々削除
npdata = np.delete(npdata, 0, 1)
npdata = np.delete(npdata, 0, 0)

npdata_random = np.delete(npdata_random, 0, 1)
npdata_random = np.delete(npdata_random, 0, 0)

#npdata = 評価行列

rui = npdata
rui_test = npdata_random

c = 0
summ = []


lis  = [i for i in range(1682*943)]


Eui = 0



#RMSEの算出
RMSE = 0
rmse_c = 0
for i in range(943):
    for j in range(1682):
        if  rui[i][j] != 0:
            RMSE = RMSE + (rui_test[i][j] -  rui[i][j])**2
            rmse_c = rmse_c + 1


RMSE = math.sqrt(RMSE / rmse_c)

print(RMSE)
#print ("time:{0}".format(end_time) + "[sec]")
print(rmse_c)




