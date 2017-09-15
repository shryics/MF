
import random
import pandas as pd
import numpy as np
import math

RMSE_list = []

user_num = 943
item_num = 1682

for cv in range(5):

    # データ読み込み
    data = pd.read_csv('ml-100k/ml-100k/u'+str(cv+1)+'.base', sep="\t", header= None)
    data_test = pd.read_csv('ml-100k/ml-100k/u'+str(cv+1)+'.test', sep="\t", header= None)

    # 評価行列の生成
    npdata = np.zeros((user_num+1, item_num+1))
    npdata_test = np.zeros((user_num+1, item_num+1))
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
    user = np.zeros((user_num, k))
    item = np.zeros((k, item_num))

    # user 乱数による初期化
    for i in range(user_num):
        for j in range(k):
            user[i][j] = random.randint(0, 1)
    # item 乱数による初期化
    for i in range(item_num):
        for j in range(k):
            item[j][i] = random.randint(0, 1)

    # trainデータ
    rui = npdata
    # testデータ
    rui_test = npdata_test

    # ハイパーパラメータの定義
    lam = 0.001  # ラムダ　正則化項の係数
    gan = 0.01   # ガンマ　学習率

    lis = [i for i in range(item_num*user_num)]
    Eui = 0
    c = 0
    summ = []

    while(c <= 5000):
        c = c + 1
        # 要素をランダムに取り出す順序決め
        rc = np.random.choice(lis, len(lis), replace=True)

    # ユーザ行列とアイテム行列の更新
    # 1行1列目から1行2列目...というように順に更新するのではなく,値をランダムに取得し,更新
        l = 0
        while (l != item_num * user_num):
            a = rc[l]
            s = a // item_num  # i
            t = a % item_num   # j
            l = l + 1
            if rui[s][t] > 0:
                Eui = rui[s][t] - np.dot(user[s, :], item[:, t])
                item[:, t] = item[:, t] + (gan / math.sqrt(c + 1)) * (Eui * user[s, :] - lam * item[:, t])
                user[s, :] = user[s, :] + (gan / math.sqrt(c + 1)) * (Eui * item[:, t] - lam * user[s, :])

        if c == 1:
            err_b = -1

    # 誤差の算出
        err = 0
        for i in range(user_num):
            for j in range(item_num):
                if rui[i][j] > 0:
                    err = err + ((rui[i, j] - np.dot(user[i, :], item[:, j])) * (rui[i, j] - np.dot(user[i, :], item[:, j])) + lam * (
                    (np.linalg.norm(item[:, j])) * (np.linalg.norm(item[:, j])) + (np.linalg.norm(user[i, :])) * (
                    np.linalg.norm(user[i, :]))))

        summ.append(err/100000)

    # 前との誤差の差が 0.000001より小さければ終了
        if abs(err/100000 - err_b) < 0.001:
            break
        err_b = err/100000

    Rmf = np.dot(user, item)

    # RMSEの算出
    RMSE_cv = 0
    rmse_c = 0
    for i in range(user_num):
        for j in range(item_num):
            if rui_test[i][j] != 0:
                RMSE_cv = RMSE_cv + (Rmf[i][j] - rui_test[i][j])**2
                rmse_c = rmse_c + 1

    RMSE_cv = math.sqrt(RMSE_cv / rmse_c)
    RMSE_list.append(RMSE_cv)

RMSE = sum(RMSE_list) / 5

print(RMSE)
