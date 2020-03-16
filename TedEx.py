#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import io
import matplotlib.pyplot as plt
import math
from scipy.spatial import distance
import ast
from typing import Callable
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import operator
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from yellowbrick.text import TSNEVisualizer
from yellowbrick.features import Manifold
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

import random
from tuner import SVM_TUNER
import flash
from numpy import linalg as LA

def create_structures():
    if os.path.isdir('temp') is False:
        os.mkdir('temp')
    if os.path.isdir('result') is False:
        os.mkdir('result')
    if os.path.isdir('data') is False:
        os.mkdir('data')

def processInputFile(filename1, filename2):
    df = pd.read_csv(filename1)
    df2 = pd.read_csv(filename2)
    df.index.name='id'
    df = pd.merge(df, df2, on='url', how='inner')

    df['talkType'] = None
    for i, row in enumerate(df['ratings']):
        ratings = ast.literal_eval(row)
        rating_dict = {}
        for rating in ratings:
            rating_dict[rating['name']] = rating['count']
        max_rating_name = max(rating_dict.items(), key=operator.itemgetter(1))[0]
        df.loc[i, 'talkType'] = max_rating_name

    quant25, quant75 = df['views'].quantile([0.25, 0.75]);
    df['popularity'] = 0;
    df['year'] = list([datetime.utcfromtimestamp(i).strftime('%Y') for i in df['published_date']])
    quant25l = []
    quant75l = []
    years = np.unique(np.array(df['year']));
    for i in years:
        filter1 = df['year'] == i;
        df_new = df.where(filter1);
        quant25l.append(df_new['views'].quantile(0.25));
        quant75l.append(df_new['views'].quantile(0.75));
    
    for i,row in enumerate(df['views']):
        yearIndex = np.where(years==str(df['year'][i]));
        if(int(row) > quant75l[yearIndex[0][0]]):
            df.loc[i,'popularity'] = "Very popular";
        elif(int(row) < quant25l[yearIndex[0][0]]):
            df.loc[i,'popularity'] = "Unpopular";
        else:
            df.loc[i,'popularity'] = "Moderately popular";    

    df = df.drop(columns=['year', 'ratings','url', 'film_date', 'main_speaker', 'name', 'related_talks', 'speaker_occupation'])

    df.loc[df['talkType'].isin(['Longwinded', 'Unconvincing', 'OK', 'Confusing', 'Obnoxious']), 'talkType'] = 'Other'
    print("\ncategory details:")
    for talkType in df['talkType'].unique():
        print(talkType, len(df.loc[df['talkType']==talkType]))

    print("\ncategory details (popularity):")
    for popularity in df['popularity'].unique():
        print(popularity, len(df.loc[df['popularity']==popularity]))

    print("\nregrission details:")
    print(df['views'].describe())

    df.to_csv("./data/processed.csv")
    df.to_pickle("./data/processed.pkl")

create_structures()
processInputFile('./data/ted_main.csv', './data/transcripts.csv')


# In[ ]:


class TEDTALK:
    #stores regression and classification results
    global met_regression, met_classification
    met_regression, met_classification = {}, {}
    def __init__(self, filename: str, clf, target_col, desc="default", classification=False, split=10):
        if '.pkl' in filename:
            self.df = pd.read_pickle(filename)
        else:
            self.df = pd.read_csv(filename)
        self.random_state = 1
        self.models = []
        self.results = []
        self.target_col = target_col
        self.columns = ['transcript', 'description', 'title', 'tags', 
                        'event', 'comments', 'duration', 'languages',
                        'num_speaker', 'published_date', self.target_col] #'speaker_occupation'
        # self.columns = ['comments', 'duration', 'languages',
        #                 'num_speaker', 'published_date', self.target_col] #'speaker_occupation'
        
        self.splits = split
        self.shuffle = True

        self.classification = classification
        self.clf = clf
        self.desc = desc
        
        
    def runCrossValidate(self):
        print('--current prediction details: ' + self.desc + ' on ' + self.target_col + '--')
        
        if self.classification:
            kf = StratifiedKFold(n_splits=self.splits, shuffle=self.shuffle, random_state=self.random_state);
        else:
            kf = KFold(n_splits=self.splits, random_state=self.random_state, shuffle=self.shuffle);
        
        df = self.df[self.columns].copy()
        
        fold = 0
        if self.classification:
            spl = kf.split(df,df[self.target_col]);
        else:
            spl = kf.split(df);
        for train_index, test_index in spl:
            df_train, df_test = df.loc[train_index], df.loc[test_index]
            df_train, df_valid = train_test_split(df_train, test_size=.1, random_state=self.random_state, shuffle=self.shuffle)
            self.models.append(MODEL(self.clf, df_train, df_valid, df_test, self.random_state, self.target_col, fold))
            fold += 1
            
    def runExpriment(self):
        for i, model in enumerate(self.models):
            model.preprocess()
            model.predict()
            print('--FOLD [' + str(i) + ']--')

        self.combine_results()

    def runExprimentSVMTuned(self):
        for i, model in enumerate(self.models):
            model.preprocess()
            model.tune()
            model.predict()
            print('--FOLD [' + str(i) + ']--')

        self.combine_results()

    def combine_results(self):
        dfs = []
        for file in os.listdir('temp/'):
            if 'test_df_' in file and '.pkl' in file:
                dfs.append(pd.read_pickle('temp/' + file))
        df = pd.concat(dfs, sort=False)
        df.to_pickle('result/' + self.desc + '.pkl')
        print('--combined_results--')

    def result(self):
        df = pd.read_pickle('result/' + self.desc + '.pkl')
        if self.classification is False:
            #Calculation of RMSE for classifiers
            met_regression[self.desc] = np.sqrt(mean_squared_error(df[self.target_col], df[self.target_col + '_pred']))
            print("RMSE:", np.sqrt(mean_squared_error(df[self.target_col], df[self.target_col + '_pred'])))
        else:
            #Calculation of F1-Score for regressors
            met_classification[self.desc] = f1_score(df[self.target_col], df[self.target_col + '_pred'], average='micro')
            print("Confusion Matrix:\n", confusion_matrix(df[self.target_col], df[self.target_col + '_pred']))
            print("Accuracy:\n", accuracy_score(df[self.target_col], df[self.target_col + '_pred']))
            print("Precision:\n", precision_score(df[self.target_col], df[self.target_col + '_pred'], average='micro'))
            print("F1 Score:\n", f1_score(df[self.target_col], df[self.target_col + '_pred'], average='micro'))


# In[ ]:


class MODEL:
    def __init__(self, clf, df_train, df_valid, df_test, random_state, target_col, fold):
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test
        self.random_state = random_state
        self.target_col = target_col
        self.fold = fold
        self.clf = clf

        # configuration of max features
        self.max_fea = [2000, 200, 100, 100, 10] # 50 for 'speaker_occupation'

        
    def text_preprocess(self, df, col, max_fea, tfer=None):
        if (tfer==None):
            tfer = TfidfVectorizer(lowercase=True, stop_words='english', use_idf=True, smooth_idf=True, 
                          max_features=max_fea, norm='l2');
            tfer.fit(df[col])
        
        col_names = list(tfer.vocabulary_.keys())
        col_names = [col + '_' + s for s in col_names]

        tuple_vectors = tfer.transform(df[col]);
        #tsne = TSNEVisualizer(decompose_by=int(min(np.shape(tuple_vectors.toarray()))-1))
        #tsne.fit(tuple_vectors,y=np.array(df[self.target_col]))
        #tsne.poof()
        
        df_temp = pd.DataFrame(tuple_vectors.toarray(), index=df.index, columns=col_names)
        
        df = pd.concat([df, df_temp], axis=1)
        df = df.drop(columns=[col])

        return df, tfer
        
    def preprocess(self):
        temp = self.df_train.iloc[0]
        cols = self.df_train.columns

        count_fea = 0
        for i, x in enumerate(temp):
            if cols[i] == self.target_col:
                continue
            if isinstance(x, str):
                self.df_train, tfer = self.text_preprocess(self.df_train, cols[i], self.max_fea[count_fea])
                self.df_valid, tfer = self.text_preprocess(self.df_valid, cols[i], self.max_fea[count_fea], tfer)
                self.df_test, tfer = self.text_preprocess(self.df_test, cols[i], self.max_fea[count_fea], tfer)
                count_fea += 1
            else:
                #normalize the numeric columns
                colminimum = self.df_train[cols[i]].min();
                colmaximum = self.df_train[cols[i]].max();
                self.df_train[cols[i]] = self.df_train[cols[i]].transform(lambda x: (x-colminimum)/(colmaximum-colminimum));
                self.df_valid[cols[i]] = self.df_valid[cols[i]].transform(lambda x: (x-colminimum)/(colmaximum-colminimum));
                self.df_test[cols[i]] = self.df_test[cols[i]].transform(lambda x: (x-colminimum)/(colmaximum-colminimum));
        
        global col_eig_train_column_nv, col_eig_train_values
        train_cov_matrix = self.df_train[self.df_train.columns[~self.df_train.columns.isin([self.target_col])]].cov()
        w, v = LA.eig(train_cov_matrix)
        
        col_eig_train = []
        for i in range(len(w)):
            if self.df_train.columns[i] != self.target_col:
                col_eig_train.append((self.df_train.columns[i],w[i]))
        
        col_eig_train_sort = sorted(col_eig_train, key = lambda x:x[1], reverse=True)
        sum_col_eig_train = np.sum([i for j,i in col_eig_train_sort[:600]])
        col_eig_train_values = np.cumsum([i*100/sum_col_eig_train for j,i in col_eig_train_sort[:600]])
        
        #Storing column headings
        col_eig_train_column_nv = [j for j,i in col_eig_train_sort[:600]]
        col_eig_train_column = col_eig_train_column_nv.copy()

        col_eig_train_column.append(self.target_col)
        
        self.df_train = self.df_train[col_eig_train_column]
        self.df_test = self.df_test[col_eig_train_column]
        self.df_valid = self.df_valid[col_eig_train_column]
        
        
        self.save_pkl()
        print('--finished preprocessing--')
        return
    
    def save_pkl(self):
        self.df_train.to_pickle('temp/train_df_' + str(self.fold) + ".pkl")
        self.df_valid.to_pickle('temp/valid_df_' + str(self.fold) + ".pkl")
        self.df_test.to_pickle('temp/test_df_' + str(self.fold) + ".pkl")

    def predict(self):
        train_x = self.df_train.loc[:, self.df_train.columns != self.target_col]
        train_y = np.ravel(self.df_train.loc[:, self.df_train.columns == self.target_col])
        test_x = self.df_test.loc[:, self.df_test.columns != self.target_col]
        test_y = self.df_test.loc[:, self.df_test.columns == self.target_col]
        
        self.clf.fit(train_x, train_y)
        pred_y = self.clf.predict(test_x)
        self.df_test[self.target_col + '_pred'] = pred_y
       
        self.save_pkl()
        
        print('--finished predicting--')
        
        
    def tune(self):
        train_x = self.df_train.loc[:, self.df_train.columns != self.target_col]
        train_y = np.ravel(self.df_train.loc[:, self.df_train.columns == self.target_col])
        tune_x = self.df_valid.loc[:, self.df_valid.columns != self.target_col]
        tune_y = self.df_valid.loc[:, self.df_valid.columns == self.target_col]
        
        tuner = SVM_TUNER()
        best_config = flash.tune_with_flash(tuner, train_x, train_y, tune_x, tune_y, random_seed=0)
        
        
        self.clf = tuner.get_clf(best_config)
        
        print('--finished tuning svm [' + str(best_config) + ']--')
        


# In[ ]:


# clf = DecisionTreeRegressor(random_state=1)
# ted = TEDTALK('/content/processed.pkl', clf, target_col='views', desc='DTRegressor')
# ted.runCrossValidate()
# ted.runExpriment()
# ted.combine_results()
# ted.result()

clf = DecisionTreeRegressor(random_state=1)
ted = TEDTALK('./data/processed.pkl', clf, target_col='views', desc='DTRegressor')
ted.runCrossValidate()
ted.runExpriment()
ted.combine_results()
ted.result()

clf = DecisionTreeClassifier(random_state=1)
ted = TEDTALK('./data/processed.pkl', clf, target_col='talkType', desc='DTClassifier', classification=True)
ted.runCrossValidate()
ted.runExpriment()
ted.combine_results()
ted.result()

clf = LinearRegression(fit_intercept=True, normalize=True);
ted = TEDTALK('./data/processed.pkl', clf, target_col='views', desc='Linear Reg')
ted.runCrossValidate()
ted.runExpriment()
ted.combine_results()
ted.result()

clf = Ridge(alpha=2.0,fit_intercept=True, normalize=True);
ted = TEDTALK('./data/processed.pkl', clf, target_col='views', desc='Ridge Reg')
ted.runCrossValidate()
ted.runExpriment()
ted.combine_results()
ted.result()

clf = SVR(C=1.0, gamma='auto', kernel='rbf');
ted = TEDTALK('./data/processed.pkl', clf, target_col='views', desc='SV Reg')
ted.runCrossValidate()
ted.runExpriment()
ted.combine_results()
ted.result()

clf = SVC(C=100.0, gamma='auto', kernel='linear');
ted = TEDTALK('./data/processed.pkl', clf, target_col='talkType', desc='SVClassifier', classification=True)
ted.runCrossValidate()
ted.runExpriment()
ted.combine_results()
ted.result()

clf = SVC(gamma='auto', kernel='linear');
ted = TEDTALK('./data/processed.pkl', clf, target_col='popularity', desc='SVClassifierTuned', classification=True)
ted.runCrossValidate()
ted.runExprimentSVMTuned()
ted.combine_results()
ted.result()

clf = RandomForestClassifier(n_estimators=100, max_depth=3,random_state=0);
ted = TEDTALK('./data/processed.pkl', clf, target_col='talkType', desc='RFClassifier', classification=True)
ted.runCrossValidate()
ted.runExpriment()
ted.combine_results()
ted.result()


# In[ ]:


# df = pd.read_pickle('./result/DTClassifier.pkl')
# df[['talkType', 'talkType_pred']]
# # df_temp = ted.models[0].df_test

# # df_temp['id'] = df_temp.index 

# # ax = df_temp.plot(x='id', y='views', kind='scatter')
# # df_temp.plot(x='id', y='views_pred', kind='scatter', ax=ax, color='r')


# # df_temp['views'].describe()
dfPkl = pd.read_pickle('./data/processed.pkl')
tempNew = dfPkl.iloc[0]
cols = dfPkl.columns;
max_fea = [2000, 200, 100, 100, 10];

j=0;
for (i,x) in enumerate(tempNew):
    if isinstance(x, str) and (cols[i] != 'popularity') and (cols[i] != 'talkType'):
        print("The column here is"+cols[i]);
        fig = plt.figure()
        ax = plt.axes(projection='3d');
        tfer = TfidfVectorizer(lowercase=True, stop_words='english', use_idf=True, smooth_idf=True, max_features=max_fea[j], norm='l2');
        j+=1;
        tfer.fit(dfPkl[cols[i]])
        tuple_vectors = tfer.transform(dfPkl[cols[i]]);
        #tsne = TSNEVisualizer(decompose_by=int(min(np.shape(tuple_vectors.toarray()))-1))
        tsne = TSNEVisualizer(ax=ax,decompose_by=int(9))
        tsne.fit(tuple_vectors,y=np.array(dfPkl['popularity']))
        tsne.poof()
        fig = plt.figure()
        tsne = TSNEVisualizer(decompose_by=int(9))
        tsne.fit(tuple_vectors,y=np.array(dfPkl['popularity']))
        tsne.poof()
j = 0;
for (i,x) in enumerate(tempNew):
    if isinstance(x, str) and (cols[i] != 'popularity') and (cols[i] != 'talkType'):
        print("The column here is"+cols[i]);
        fig = plt.figure()
        ax = plt.axes(projection='3d');
        tfer = TfidfVectorizer(lowercase=True, stop_words='english', use_idf=True, smooth_idf=True, max_features=max_fea[j], norm='l2');
        j+=1;
        tfer.fit(dfPkl[cols[i]])
        tuple_vectors = tfer.transform(dfPkl[cols[i]]);
        #tsne = TSNEVisualizer(decompose_by=int(min(np.shape(tuple_vectors.toarray()))-1))
        tsne = TSNEVisualizer(ax=ax,decompose_by=int(9))
        tsne.fit(tuple_vectors,y=np.array(dfPkl['talkType']))
        tsne.poof()
        tsne = TSNEVisualizer(decompose_by=int(9))
        tsne.fit(tuple_vectors,y=np.array(dfPkl['talkType']))
        tsne.poof()


# In[ ]:


#PCA Plot - All components
plt.clf()
fig = plt.figure(figsize=(5,5))
ax1 = fig.add_subplot(111)
ax1.scatter(col_eig_train_column_nv, abs(col_eig_train_values), c='b', label='Training Accuracy')
ax1.set_xlabel('Components', fontsize=16)
ax1.set_ylabel('Variance', fontsize=16)
plt.legend(loc='upper left')
plt.title("Variance vs Components", fontsize=20)
plt.xticks([])
#plt.xticks(col_eig_train_column_nv[0])
plt.show()

#PCA Plot - First 10 components
plt.clf()
fig = plt.figure(figsize=(5,5))
ax1 = fig.add_subplot(111)
ax1.scatter(col_eig_train_column_nv[:10], abs(col_eig_train_values[:10]), c='b', label='Training Accuracy')
ax1.set_xlabel('Components', fontsize=16)
ax1.set_ylabel('Variance', fontsize=16)
plt.legend(loc='upper left')
plt.title("Variance vs Components", fontsize=20)
for xy in zip(col_eig_train_column_nv[:10], np.round(abs(col_eig_train_values[:10]),2)):
    ax1.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
plt.xticks([])
#plt.xticks(col_eig_train_column_nv[0])
plt.show()


# In[ ]:


#Bar graph for Classification and Regression
plt.rc('xtick', labelsize=14)
plt.bar(met_classification.keys(), met_classification.values(),width = 0.7, color = ['red', 'green','blue'])
plt.xlabel('Classification Algorithms', fontsize=18)
plt.ylabel('F-Score', fontsize=18)
plt.suptitle('Classification Comparison for popularity label')
plt.show()

plt.bar(met_regression.keys(), met_regression.values(), color=['y','black','maroon','purple'])
plt.suptitle('Regression Plotting')
plt.xlabel('Regression Algorithms', fontsize=18)
plt.ylabel('Root Mean Square Error', fontsize=18)
plt.show()

