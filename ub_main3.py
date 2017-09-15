
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time

RMSE_list = []

for cv in range(5):
    #データ読み込み
    d_train = pd.read_csv('ml-100k/ml-100k/u'+str(cv+1)+'.base', sep="\t", header= None)
    d_test = pd.read_csv('ml-100k/ml-100k/u' + str(cv + 1) + '.test', sep="\t", header=None)


    #評価行列の生成
    npdata = np.zeros((944, 1683))
    npdata_test = np.zeros((944, 1683))
    for i in range(len(d_train)):
        npdata[int(d_train[0][i])][d_train[1][i]] = d_train[2][i]
    for i in range(len(d_test)):
        npdata_test[int(d_test[0][i])][d_test[1][i]] = d_test[2][i]
    #npdata の　一行目　一列目　其々削除
    npdata = np.delete(npdata, 0, 1)
    npdata = np.delete(npdata, 0, 0)
    npdata_test = np.delete(npdata_test, 0, 1)
    npdata_test = np.delete(npdata_test, 0, 0)

    rui = npdata
    ruiori = npdata
    rui_test = npdata_test


    for k in range(943):
        print(k)
        #print(rui[k].shape)



        u_avg =np.average(rui[k])
        pia_ = []
        j_ = []

        #類似度の算出
        for j in range(943-1):
            ab = 0
            root_l = 0
            root_r = 0
            u_avg2 = np.average(ruiori[j+1])
            skip = 0
            for i in range(1682):
                cir  = ruiori[k][i] - u_avg
                if k == j:
                    skip = 1
                cir2 = ruiori[j+skip][i] - u_avg2
                ab = ab + (cir) * (cir2)
                root_l = root_l + (cir)**2
                root_r = root_r + (cir2)**2

            pia = ab / ( math.sqrt(root_l) * math.sqrt(root_r) )
            if pia > 0.2:

                pia_.append(pia)
                j_.append(j)
                #print(pia,j)


        #アイテムの評価が無い(0)の場所へ類似ユーザ(ピアソン相関係数0.2以上)の同アイテムの評価値の加重平均を算出
        for i in range(1682):
            c = 0
            total_u = 0
            total_d = 0
            if ruiori[k][i] == 0:
                for j in range(943-1):
                    if j == j_[c]:
                        total_u = total_u + (np.average(ruiori[j+1]) - ruiori[j+1][i]) * pia_[c]
                        total_d = total_d + pia_[c]
                        if c != len(j_)-1:
                            c = c + 1
                rui[k][i] = u_avg + (total_u / (total_d))


    ###以上でrui 0の穴埋めが完了
    #ruiがbase
    #rui_testで検証
    ###以下から testデータを用いてRMSEを算出



    #RMSEの算出
    RMSE_cv = 0
    rmse_c = 0
    for i in range(943):
        for j in range(1682):
            if  rui_test[i][j] != 0:
                RMSE_cv = RMSE_cv + (rui[i][j] -  rui_test[i][j])**2
                rmse_c = rmse_c + 1


    RMSE_cv = math.sqrt(RMSE_cv / rmse_c)

    RMSE_list.append(RMSE_cv)
    print(RMSE_cv)
    #print ("time:{0}".format(end_time) + "[sec]")
    print(rmse_c)

RMSE = sum(RMSE_list) / 5

print(RMSE)