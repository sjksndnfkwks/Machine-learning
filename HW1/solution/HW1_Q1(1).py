__author__ = 'XF'
__date__ = '2024/09/21'


'''
    linear regression.
'''

import numpy as np
from sklearn.linear_model import LinearRegression


def get_data(train_data_path, test_data_path):

    train_data = np.loadtxt(train_data_path)
    test_data = np.loadtxt(test_data_path)
    print(f'load train data: {train_data.shape}')
    print(f'load test_data: {test_data.shape}')
    return train_data[:, :-1], train_data[:, -1], test_data


def linear_regresssion(X_train, Y_train, X_test):

    print(f'############################# Using Sklearn #############################')
    model = LinearRegression()
    model.fit(X_train, Y_train)

    print(f'Parameters ====================')
    print(f'w_0: {model.intercept_}')
    for i in range(len(model.coef_)):
        print(f'w_{i+1}: {model.coef_[i]}')
    print(f'Predictions ===================')
    print(model.predict(X_test))

    print(f'############################## By Yourself ##############################')
    new_X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)
    coefs = np.matmul(np.matmul(np.linalg.inv(np.matmul(new_X_train.T, new_X_train)), new_X_train.T), Y_train)
    print(f'Parameters ====================')
    for i in range(len(coefs)):
        print(f'w_{i}: {coefs[i]}')
    print(f'Predictions ===================')
    predict_y = np.matmul(np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1), coefs)
    print(predict_y)



if __name__ == '__main__':

    train_data_path = '.\\data\\linear_regression_train.txt'
    test_data_path = '.\\data\\linear_regression_test.txt'

    X_train, Y_train, X_test = get_data(train_data_path, test_data_path)

    linear_regresssion(X_train, Y_train, X_test)

