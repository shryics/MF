
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time

RMSE_list = []

for cv in range(5):

    # データ読み込み
    data = pd.read_csv('ml-100k/ml-100k/u'+str(cv+1)+'.base', sep="\t", header= None)
    data_test = pd.read_csv('ml-100k/ml-100k/u'+str(cv+1)+'.test', sep="\t", header= None)

    # 評価行列の生成
    npdata = np.zeros((944, 1683))
    npdata_test = np.zeros((944, 1683))
    for i in range(len(data)):
        npdata[int(data[0][i])][data[1][i]] = data[2][i]
    for i in range(len(data_test)):
        npdata_test[int(data_test[0][i])][data_test[1][i]] = data_test[2][i]

    # npdata の　一行目　一列目　其々削除
    npdata = np.delete(npdata, 0, 1)
    npdata = np.delete(npdata, 0, 0)
    npdata_test = np.delete(npdata_test, 0, 1)
    npdata_test = np.delete(npdata_test, 0, 0)

    # npdata = 評価行列, user = ユーザ行列, item = アイテム行列
    k = 30  # ユーザ、アイテム行列の幅
    user = np.zeros((943, k))
    item = np.zeros((k, 1682))

    # user 乱数による初期化
    for i in range(943):
        for j in range(k):
            user[i][j] = random.randint(0, 1)
    # item 乱数による初期化
    for i in range(1682):
        for j in range(k):
            item[j][i] = random.randint(0, 1)

    # trainデータ
    rui = npdata
    # testデータ
    rui_test = npdata_test

    # Eui = rui - np.dot(user, item) #要らない?
    c = 0
    summ = []

    # ハイパーパラメータの定義
    lam = 0.001  # ラムダ　正則化項の係数
    gan = 0.01   # ガンマ　学習率

    lis = [i for i in range(1682*943)]
    Eui = 0

    # start = time.time()
    while(c <= 5000):
        c = c + 1
        # 要素をランダムに取り出す順序決め
        rc = np.random.choice(lis, len(lis), replace=True)

    # ユーザ行列とアイテム行列の更新
        l = 0
        while (l != 1682 * 943):
            a = rc[l]
            s = a // 1682  # i
            t = a % 1682   # j
            l = l + 1
            if rui[s][t] > 0:
                Eui = rui[s][t] - np.dot(user[s, :], item[:, t])
                item[:, t] = item[:, t] + (gan / math.sqrt(c + 1)) * (Eui * user[s, :] - lam * item[:, t])
                user[s, :] = user[s, :] + (gan / math.sqrt(c + 1)) * (Eui * item[:, t] - lam * user[s, :])

        if c == 1:
            err_b = -1

    # 誤差の算出
        err = 0
        for i in range(943):
            for j in range(1682):
                if rui[i][j] > 0:
                    err = err + ((rui[i, j] - np.dot(user[i, :], item[:, j])) * (rui[i, j] - np.dot(user[i, :], item[:, j])) + lam * (
                    (np.linalg.norm(item[:, j])) * (np.linalg.norm(item[:, j])) + (np.linalg.norm(user[i, :])) * (
                    np.linalg.norm(user[i, :]))))

        summ.append(err/100000)
        print(err/100000, abs(err/100000 - err_b), c)

    # 前との誤差の差が 0.000001より小さければ終了
        if abs(err/100000 - err_b) < 0.001:
            break
        err_b = err/100000

    # end_time = time.time() - start


    #plt.plot(summ)
    #plt.show()

    print("min     max")
    print(np.min(np.dot(user, item)), np.max(np.dot(user, item)))
    print("min  u   max")
    print(np.min(user), np.max(user))
    print("min  i   max")
    print(np.min(item), np.max(item))

    print(item.shape)
    print(user.T.shape)

    Rmf = np.dot(user, item)

    # RMSEの算出
    RMSE_cv = 0
    rmse_c = 0
    for i in range(943):
        for j in range(1682):
            if rui_test[i][j] != 0:
                RMSE_cv = RMSE_cv + (Rmf[i][j] - rui_test[i][j])**2
                rmse_c = rmse_c + 1

    RMSE_cv = math.sqrt(RMSE_cv / rmse_c)

    RMSE_list.append(RMSE_cv)
    print(RMSE_cv)
    #print ("time:{0}".format(end_time) + "[sec]")
    print(rmse_c)

RMSE = sum(RMSE_list) / 5

print(RMSE)
