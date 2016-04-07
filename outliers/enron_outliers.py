#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop('TOTAL',0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
### your code below

# max_sals = []
# for k,v in data_dict.items():
#     if data_dict[k]['salary'] != 'NaN':
#         max_sals.append((k,data_dict[k]['salary']))
# print sorted(max_sals,key=lambda x:x[1],reverse=True)

for k,v in data_dict.items():
    if data_dict[k]['bonus'] != 'NaN':
        if data_dict[k]['bonus'] >= 5000000:
            print (k,data_dict[k]['bonus'],data_dict[k]['salary'])


for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
