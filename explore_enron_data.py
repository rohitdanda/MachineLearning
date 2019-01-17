import pandas as pd
import re
from final_project import poi_email_addresses
import numpy as np

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
enron_data = pd.read_pickle("final_project/final_project_dataset_unix.pkl")
file = open("final_project/poi_names.txt")
data = file.read()
listofdata = data.splitlines()

poi_all = 0
with open("final_project/poi_names.txt") as f:
	content = f.readlines()
for line in content:
	if re.match(r'\((y|n)\)', line):
		poi_all += 1

print("All POI:", poi_all)



enron_keyPOIPayment = dict((k,enron_data[k]['total_payments']) for k in ("LAY KENNETH L", "SKILLING JEFFREY K", "FASTOW ANDREW S"))
max_earner = max(enron_keyPOIPayment, key=enron_keyPOIPayment.get)
print( "Largest total payment earner and payment:", max_earner, enron_keyPOIPayment[max_earner])
poi_dataset = 45
salaries_available = 0
emails_available = 0
total_payments_unavailable = 0
total_payments_unavailable_poi = 0
for name in enron_data:
	if not np.isnan(float(enron_data[name]['salary'])):
		salaries_available += 1
	if enron_data[name]['email_address'] != "NaN":
		emails_available += 1
	if np.isnan(float(enron_data[name]['total_payments'])):
		total_payments_unavailable += 1
		if enron_data[name]['poi']:
			total_payments_unavailable_poi += 1

print(
"Salaries available:", salaries_available)
print(
"Emails available:", emails_available)
print(
"NaN for total payment and percentage:", total_payments_unavailable, float(total_payments_unavailable) / len(
	enron_data) * 100)
print(
"NaN for total payment of POI and percentage:", total_payments_unavailable_poi, float(
	total_payments_unavailable_poi) / poi_dataset * 100)


