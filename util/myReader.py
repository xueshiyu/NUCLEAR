#！/usr/bin/env python
#-*- coding: utf-8 -*-
#@Author:Shiyu Xue

from pandas import read_csv,read_excel,DataFrame
import re
import numpy as np

dataset = read_csv('../data/velu-t.csv')
# print(dataset)
pattern = re.compile(r'([\S]+)')
dataset = np.array(dataset).astype(str)
data = []

for x in dataset:
    result=pattern.findall(x[0])
    data.append(result)
data = DataFrame(data)
data.to_excel('../data/velu.xls')



