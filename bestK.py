# loss func gives MAPE value for each K
# each which value of K gives the least loss

def findOverallLoss(ypreds, Y):
    err = abs(ypreds - Y)
    MAPE = err.mean(axis=1)
    return MAPE

def most_common_k(arr):
    counter = 0
    num = arr[0]
    for i in arr:
        curr_freq = arr.count(i)
        if(curr_freq > counter):
            counter = curr_freq
            num = i

    return num

def oneRow(row, knnout, Y):
    x = (knnout.iloc[row, :].values == Y.astype(float).values)
    return len(x[x == True]) / len(tcY)


def bestK(X, Y, maxK, maxRow):
    def kNN(X, Y, newx, k, regress=True, allK=False, leave1out=False, scaleX=True, scaler='standard'):
        import warnings
        warnings.filterwarnings('ignore')

        import numpy as np
        import pandas as pd

        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neighbors import KNeighborsRegressor

        from sklearn.preprocessing import StandardScaler
        # from sklearn.preprocessing import MinMaxScaler
        from sklearn.preprocessing import RobustScaler

        from statistics import mean
        from statistics import mode
        from collections import Counter

        def kNNtype(neighbs, regress):
            if regress:
                knn = KNeighborsRegressor(n_neighbors=neighbs)
            else:
                knn = KNeighborsClassifier(n_neighbors=neighbs)
            return knn

        if scaler != 'standard':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        # Update: for row subsets/test sets in hw; fitting the scaling function should be done separate,
        # so the same can be applied to train and test data (or X and newx)
        if scaleX == True:
            # scale should be fit to X/train
            scaler.fit(X)
            newx = pd.DataFrame(scaler.transform(newx))
            X = pd.DataFrame(scaler.transform(X))

        knn_all = pd.DataFrame()
        if allK == True:
            if leave1out == True:
                nn_all = []
                for j in list(newx.index.values.tolist()):
                    # for j in list(Y.index.values.tolist()):
                    knn_row = []
                    knn = kNNtype(k + 1, regress)
                    knn.fit(X, Y)
                    test = pd.DataFrame(newx.loc[j, :])
                    nn = knn.kneighbors(test.T)[1][0]
                    for i in range(2, k + 1):
                        nn1 = nn[1:i]  # leave one out
                        test = list(Y.iloc[nn1])
                        if regress:
                            test = mean(test)
                        else:
                            c = Counter(test)
                            l = list(c.values())
                            ind = l.index(max(c.values()))
                            test = list(c.keys())[ind]
                            # count number of times the max class occurs and if there is a tie
                            # choose the second class with the max if index is even
                            if (l.count(max(l))) > 1 and (j % 2 != 0):
                                l[ind] = 0
                                ind = l.index(max(c.values()))
                                test = list(c.keys())[ind]

                        knn_row.append(test)
                    knn_row = pd.DataFrame(knn_row)
                    knn_all = [knn_all, knn_row]
                    knn_all = pd.concat(knn_all, axis=1, ignore_index=True)
                    nn_all.append(list(nn1))
                nn_all = np.array(nn_all)
            else:
                for i in range(1, k + 1):
                    knn = kNNtype(i, regress)
                    knn.fit(X, Y)
                    test = knn.predict(newx)
                    knn_row = pd.DataFrame(test).T
                    knn_all = [knn_all, knn_row]
                    knn_all = pd.concat(knn_all, axis=0, ignore_index=True)
                nn_all = knn.kneighbors(newx)[1]
        else:
            if leave1out == True:
                knn_row = []
                for j in list(newx.index.values.tolist()):
                    # for j in list(Y.index.values.tolist()):
                    knn = kNNtype(k, regress)
                    knn.fit(X, Y)
                    test = pd.DataFrame(newx.loc[j, :])
                    nn = knn.kneighbors(test.T)[1][0]
                    nn1 = nn[1:len(nn)]

                    test = list(Y.iloc[nn1])
                    if regress:
                        test = mean(test)
                    else:
                        c = Counter(test)
                        l = list(c.values())
                        ind = l.index(max(c.values()))
                        test = list(c.keys())[ind]
                        # count number of times the max class occurs and if there is a tie
                        # choose the second class with the max if index is even
                        if (l.count(max(l))) > 1 and (j % 2 != 0):
                            l[ind] = 0
                            ind = l.index(max(c.values()))
                            test = list(c.keys())[ind]

                    knn_row.append(test)
                knn_all = pd.DataFrame(knn_row).T
                nn_all = nn1
            else:
                knn = kNNtype(k, regress)
                knn.fit(X, Y)
                test = knn.predict(newx)
                knn_all = pd.DataFrame(test)
                nn_all = knn.kneighbors(newx)[1]

        return knn_all, nn_all


    arr = []

    for rows in range(len(X[:maxRow])):
        all_preds, nn = kNN(X[:maxRow],Y[:maxRow],X.iloc[rows:rows+1,:],maxK,allK=True, leave1out=True, scaler='robust')
        loss = findOverallLoss(all_preds, Y[:maxRow])
        minimum = min(loss)
        arr.append(loss.index[loss == minimum].tolist()[0] + 2)

    print(min(loss))
    print(arr)
    k = most_common_k(arr)
    print(k)
    return k

def main():

# Data Set Path and Set-up
    import pandas as pd

    path = 'C:/Users/Aman/PycharmProjects/Homework0/day1.csv'
    data = pd.read_csv(path)
  #  print(data.shape)
    row, col = data.shape
  # print(data.head())

# Parsing Data Set into X and Y
    X = data.loc[:, ['workingday', 'temp', 'atemp', 'hum', 'windspeed', ]]
    Y = data.tot

    bestK(X, Y, 10, 50)

if __name__ == "__main__":
    main()

