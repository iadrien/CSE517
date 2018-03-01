# -*- coding: utf-8 -*-
"""
Application Project

@author: Adrien Xie
"""

import csv
 
data = open("parkinsons_updrs.data", "r")
reader = csv.reader(data, delimiter=',', quotechar=',',quoting=csv.QUOTE_MINIMAL)
    
with open('trainData.csv', 'wb') as csvfile:
    
    writer = csv.writer(csvfile)
    
    for i in reader:
        writer.writerow(i)
 
print("Writing complete")