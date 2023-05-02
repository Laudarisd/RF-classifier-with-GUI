import csv
import math
import os
import time
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from time import sleep
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from collections import OrderedDict
warnings.filterwarnings("ignore")
# import dataset and merge zone-2, and dense_ground

if not os.path.exists('./collision'):
    os.makedirs('./collision')


data = pd.read_csv('./zone1/1.csv')
data.rename(columns={"omegax": "\u03A9x", "omegay": "\u03A9y", "omegaz": "\u03A9z", "v_acc":"acc", "v_mag_omega":"|\u03A9|",
                    "v_mag_v":"|v|", "v_mag_f":"|f|" }, inplace=True)
# original_2 = pd.read_csv('./zone2/2.csv')
# original_2.rename(columns={"omegax": "\u03A9x", "omegay": "\u03A9y", "omegaz": "\u03A9z", "v_acc":"acc", "v_mag_omega":"|\u03A9|",
#                     "v_mag_v":"|v|", "v_mag_f":"|f|" }, inplace=True)



# data = pd.concat([original_2, original_2], axis=0)
#sort df in ascending order by column z
data.sort_values(by='z', inplace=True, axis=0)


initial_zone = data[(data['z'] > 3.6) & (data['z'] < 4.1)]
cone_zone =    data[(data['z'] > 1.9) & (data['z'] < 3)]
fall_zone =    data[(data['z'] > 0.8) & (data['z'] < 1.2)]
ground_zone =  data[ data['z'] < 0.4]

df = pd.concat([initial_zone, fall_zone], axis=0)
df.rename(columns={"acc": "a"}, inplace=True)
plt.figure(figsize=(10, 10))
sns.scatterplot(x='x', y='z', hue='radius', data=df, palette='Set1')
plt.savefig('./collision/collision.png', dpi=300)
plt.show()

print(df['type'].value_counts())
print(df.isna().sum())

#if radius is 0.03 then it change type column to 0 and if radius is 0.02 then it change type column to 1
df['type'] = df['radius'].apply(lambda x: 0 if x == 0.03 else 1)
#rename columns as we want
df.rename(columns={"omegax": "\u03A9x", "omegay": "\u03A9y", "omegaz": "\u03A9z", "v_a":"acc"}, inplace=True)

X = df.drop(['TIMESTEP', 'id', 'type', 'mass', 'radius'], axis=1)
X.fillna(X.mean(), inplace=True)
y = df['type']

print(X.columns)
print(X.isna().sum())
print(y.value_counts())


class  RandomForestTraining():
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        print("Training features:", self.X_train.columns)
        return self.X_train, self.X_test, self.y_train, self.y_test
    def model(self):
        print("######################: Random Forest Training Is Started :######################")
        self.rf_random = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, oob_score=True, verbose=1, n_jobs=-1)
        self.rf_random.fit(self.X_train, self.y_train)
        n_estimators = 100

    def evaluate_model(self):
        self.y_pred = self.rf_random.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        print("\n \n \nAccuracy from test data:{}".format(self.accuracy))
        self.confusion_matrix = confusion_matrix(self.y_test, self.y_pred)
        self.classification_report = classification_report(self.y_test, self.y_pred)
        print(self.classification_report)
    def f1_score(self):
        self.f1 = f1_score(self.y_test, self.y_pred, average='weighted')
        print("F1 score:{}".format(self.f1))
    def actaul_predicted(self):
        self.actual_predicted = pd.DataFrame({'Actual': self.y_test, 'Predicted': self.y_pred})
        print(self.actual_predicted)
    def important_features(self):
        #self.feature_importances = pd.DataFrame(self.rf_random.feature_importances_, index = self.X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
        self.feature_importances = pd.Series(self.rf_random.feature_importances_, index=self.X_train.columns).sort_values(ascending=False)
        print(self.feature_importances)
        #plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x=self.feature_importances, y=self.feature_importances.index, orient='h', color='b', alpha=0.7)
        plt.xlabel('Feature Importance Score', fontsize=30)
        plt.ylabel('Features', fontsize=30 )
        #increase font size in xtick
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        #plt.title("Visualizing Important Features")
        plt.savefig('./collision/feature_importance.png', dpi=300)
        plt.savefig('./collision/feature_importance.pdf', dpi=300)
    def save_result(self):
        save_dir= './collision/'
        with open(save_dir + 'rf.log', 'a') as f:
            f.write("######### Results from Random Forest Classifier ##########\n")
            f.write("############# Accuracy #############\n")
            f.write(str(self.accuracy))
            f.write("\n")
            f.write("############# F1 Score #############\n")
            f.write(str(self.f1))
            f.write("\n")
            f.write("############# Confusion Matrix #############\n")
            f.write(str(self.confusion_matrix))
            f.write("\n")
            f.write("############# Classification Report #############\n")
            f.write(str(self.classification_report))
            f.write("\n")
            f.write("############# Actual vs Predicted #############\n")
            f.write(str(self.actual_predicted))
            f.write("\n")
            f.write("############# Important Features #############\n")
            f.write(str(self.feature_importances))
            f.write("\n")
            f.write("################################################\n")
    def save_result_csv(self):
        self.lst = [len(self.feature_importances), round(self.f1, 3)]
        with open('./collision/result.csv', 'a') as f:
            write = csv.writer(f) 
            write.writerow(self.lst)
        print("f1score:{}".format(self.f1))
    def run(self):
        self.split_data()
        self.model()
        self.evaluate_model()
        self.f1_score()
        self.actaul_predicted()
        self.important_features()
        self.save_result()
        self.save_result_csv()
T = RandomForestTraining(X, y)
T.run()
