import random

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# cite Nair, Vivek, et al. "Finding faster configurations using FLASH." IEEE Transactions on Software Engineering (2018).

class SVM_TUNER:
    def __init__(self, fold_num=0):
        self.C_VALS = [1, 50]
        self.KERNELS = ['rbf', 'linear', 'sigmoid', 'poly']
        self.GAMMAS = [0, 1]
        self.COEF0S = [0, 1]
        self.enc = None
        random.seed(fold_num)

        self.label_coding()
        self.default_config = (1.0, 'rbf', 1.0, 0)

    def generate_param_combinaions(self):
        c = random.uniform(self.C_VALS[0], self.C_VALS[1])
        kernel = random.choice(self.KERNELS)
        gamma = random.uniform(self.GAMMAS[0], self.GAMMAS[1])
        coef0 = random.uniform(self.COEF0S[0], self.COEF0S[1])

        return c, kernel, gamma, coef0

    def label_coding(self):
        enc = LabelEncoder()
        enc.fit(self.KERNELS)
        self.enc = enc

    def label_transform(self, val):
        arr_list = self.enc.transform([val])
        return float(arr_list.tolist()[0])

    def label_reverse_transform(self, val):
        arr_list = self.enc.inverse_transform([int(val)])
        return arr_list.tolist()[0]

    def generate_param_pools(self, size):
        list_of_params = [self.generate_param_combinaions() for x in range(size)]
        return list_of_params

    def get_clf(self, configs):
        clf = SVC(C=configs[0], kernel=configs[1], gamma=configs[2],
                                     coef0=configs[3], random_state=0)
        return clf

    def transform_to_numeric(self, x):
        return x[0], self.label_transform(x[1]), x[2], x[3]

    def reverse_transform_from_numeric(self, x):
        return x[0], self.label_reverse_transform(x[1]), x[2], x[3]