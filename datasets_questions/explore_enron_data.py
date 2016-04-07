#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
total_people = len(enron_data)
total_features = sum([len(v) for k,v in enron_data.items()])/total_people
total_pois = sum([1 for k,v in enron_data.items() if enron_data[k]['poi']])

colwell_wesley = enron_data['COLWELL WESLEY']
jeffrey_skilling_eso = enron_data['SKILLING JEFFREY K']['exercised_stock_options']
jeffrey_skilling = enron_data['SKILLING JEFFREY K']
kenneth_lay = enron_data['LAY KENNETH L']
andrew_fastow = enron_data['FASTOW ANDREW S']
top_employee_list = [jeffrey_skilling,kenneth_lay,andrew_fastow]

# salary = 0
# emails = 0
# for k,v in enron_data.items():
#     if enron_data[k]['salary'] == 'NaN':
#         pass
#     else:
#         salary+=1
#
# for k,v in enron_data.items():
#     if enron_data[k]['email_address'] == 'NaN':
#         pass
#     else:
#         emails +=1
# print salary,emails

# no_payments = 0
no_payments = sum([1 for k,v in enron_data.items() if enron_data[k]['total_payments'] == 'NaN'])
# for k,v in enron_data.items():
#     if enron_data[k]['total_payments'] == 'NaN':
#         no_payments+=1
#     else:
#         pass
# print float(no_payments)/float(total_people)
no_payments_poi = float(sum([1 for k,v in enron_data.items() if enron_data[k]['poi'] and enron_data[k]['total_payments'] == 'NaN']))/float(total_pois)
print (total_pois + 10), (no_payments_poi + 10)
