#！/usr/bin/env python
#-*- coding: utf-8 -*-
#@Author:Shiyu Xue


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
pd.options.display.expand_frame_repr = False

dataset = pd.read_excel('../data/press.xls',header=0, index_col=0)
# print(dataset)
# loads the 'pollution.csv' and treat each column as a separate subplot

values = dataset.values
values = values[1:,:]
print(values[:,2])
values = ss.fit_transform(values)
print(values[:,2])
# print(values[:,3])
# print(values)
groups = [120]
# print(values[1:,3])
# print(dataset.columns[3])
i = 1

plt.figure()
for group in groups:
    p = plt.subplot(len(groups), 1, i)
    p.plot(values[:, group])
    plt.title(dataset.columns[group], y=1, loc='right')
    # p.axis([0.0,30,-40,40])
    i += 1
plt.show()
