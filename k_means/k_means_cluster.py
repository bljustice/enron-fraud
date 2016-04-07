#!/usr/bin/python

"""
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit




def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it!
data_dict.pop("TOTAL", 0)

max_sals = float(max(data_dict[k]['salary'] for k,v in data_dict.items() if data_dict[k]['salary'] != 'NaN'))
min_sals = float(min(data_dict[k]['salary'] for k,v in data_dict.items() if data_dict[k]['salary'] != 'NaN'))
max_exs = float(max(data_dict[k]['exercised_stock_options'] for k,v in data_dict.items() if data_dict[k]['exercised_stock_options'] != 'NaN'))
min_exs = float(min(data_dict[k]['exercised_stock_options'] for k,v in data_dict.items() if data_dict[k]['exercised_stock_options'] != 'NaN'))
print (200000-min_sals)/(max_sals-min_sals)
print (1000000-min_exs)/(max_exs-min_exs)


# ex_items = []
# for k,v in data_dict.items():
#     ex = data_dict[k]['exercised_stock_options']
#     if ex:
#         if ex=='NaN':
#             pass
#         else:
#             ex_items.append(ex)
#     else:
#         pass
# print max(ex_items),min(ex_items)

# sal_items = []
# for k,v in data_dict.items():
#     sal = data_dict[k]['salary']
#     if sal:
#         if sal=='NaN':
#             pass
#         else:
#             sal_items.append(sal)
#     else:
#         pass
# print max(sal_items),min(sal_items)

### the input features we want to use
### can be any key in the person-level dictionary (salary, director_fees, etc.)
feature_1 = "salary"
feature_2 = "exercised_stock_options"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list)
poi, finance_features = targetFeatureSplit(data)


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2 in finance_features:
    plt.scatter(f1,f2)
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
from sklearn.cluster import KMeans

clf = KMeans(n_clusters=2)
clf.fit(data)
pred = clf.predict(data)



### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"
