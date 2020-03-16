import random

# cite Nair, Vivek, et al. "Finding faster configurations using FLASH." IEEE Transactions on Software Engineering (2018).

# from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from tuner import SVM_TUNER
import numpy as np
from sklearn.metrics import f1_score

BUDGET = 10
POOL_SIZE = 10000
INIT_POOL_SIZE = 10

# def tune_dt(x_train, y_train, project_name):
#     tuner = DT_TUNER()
#     sss = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=0)
#     for train_index, tune_index in sss.split(x_train, y_train):
#         x_train_flash, x_tune_flash = x_train[train_index], x_train[tune_index]
#         y_train_flash, y_tune_flash = y_train.iloc[train_index], y_train.iloc[tune_index]
#         best_conf = tune_with_flash(tuner, x_train_flash, y_train_flash, x_tune_flash, y_tune_flash, project_name,
#                                     random_seed=1)

#     return best_conf


def tune_with_flash(tuner, x_train, y_train, x_tune, y_tune, random_seed=0):
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    random.seed(random_seed)

    # print("DEFAULT F1: " + str(measure_fitness(tuner, x_train, y_train, x_tune, y_tune, tuner.default_config)))

    this_budget = BUDGET

    # Make initial population
    param_search_space = tuner.generate_param_pools(POOL_SIZE)

    # Evaluate initial pool
    evaluted_configs = random.sample(param_search_space, INIT_POOL_SIZE)
    #param_search_space = list(set(param_search_space) - (set(evaluted_configs)))

    f_scores = [measure_fitness(tuner, x_train, y_train, x_tune, y_tune, configs) for configs in evaluted_configs]

    # print("F Score of init pool: " + str(f_scores))

    # hold best values
    ids = np.argsort(f_scores)[::-1][:1]
    best_f = f_scores[ids[0]]
    best_config = evaluted_configs[ids[0]]

    # converting str value to int for CART to work
    evaluted_configs = [tuner.transform_to_numeric(x) for x in evaluted_configs]
    param_search_space = [tuner.transform_to_numeric(x) for x in param_search_space]

    # number of eval
    eval = 0
    while this_budget > 0:
        cart_model = DecisionTreeRegressor(random_state=0)

        cart_model.fit(evaluted_configs, f_scores)

        next_config_id = acquisition_fn(param_search_space, cart_model)
        next_config = param_search_space.pop(next_config_id)

        next_config_normal = tuner.reverse_transform_from_numeric(next_config)

        next_f = measure_fitness(tuner, x_train, y_train, x_tune, y_tune, next_config_normal)

        if np.isnan(next_f) or next_f == 0:
            continue

        f_scores.append(next_f)
        evaluted_configs.append(next_config)

        if isBetter(next_f, best_f):
            best_config = next_config_normal
            best_f = next_f
            this_budget += 1
            # print("new F: " + str(best_f) + " budget " + str(this_budget))
        this_budget -= 1
        eval += 1

    # print("Eval: " + str(eval))

    return best_config


def acquisition_fn(search_space, cart_model):
    predicted = cart_model.predict(search_space)

    ids = np.argsort(predicted)[::-1][:1]
    val = predicted[ids[0]]
    return ids[0]

def isBetter(new, old):
    return old < new

def measure_fitness(tuner, x_train, y_train, x_tune, y_tune, configs):
    clf = tuner.get_clf(configs)
    clf.fit(x_train, np.ravel(y_train))
    y_pred = clf.predict(x_tune)
    return f1_score(y_tune, y_pred, average='micro')