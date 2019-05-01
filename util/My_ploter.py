#！/usr/bin/env python
#-*- coding: utf-8 -*-
#@Author:Shiyu Xue


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
ss = MinMaxScaler(feature_range=(0, 1))
pd.options.display.expand_frame_repr = False

dataset = pd.read_excel('../data/国电头/velu.xls',header=0, index_col=0)
# print(dataset)
# loads the 'pollution.csv' and treat each column as a separate subplot
columns_dict = {
    "0":"入口温度",
    "1":"出口温度",
    "2":"出口过冷度",
    "3":"稳压器压力",
    "4":"稳压器水位",
    "5":"174流量",
    "6":"274流量",
    "7":"374流量",
    "8":"474流量",
    "9":"完整SG水位",
    "10":"破损SG水位",
    "11":"完整SG蒸汽压力",
    "12":"破损SG蒸汽压力",
    "13":"完整SG蒸汽温度",
    "14":"破损SG蒸汽温度",
    "15":"完整SG蒸汽流量",
    "16":"破损SG蒸汽流量",
    "17":"功率",
    "18":"上充",
    "19":"下泄",
    "20":"冷却剂平均温度",
    "21":"破口流量"
}
values = dataset.values
values = values[1:-3,:-2]
# print(values[1:,3])
values = ss.fit_transform(values)
groups = [30]

# print(dataset.columns[3])
i = 1
plt.figure()
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
for group in groups:
    # print(columns_dict[str(group)])
    p = plt.subplot(len(groups), 1, i)
    p.plot(values[:, group])
    plt.title(dataset.columns[group], loc='right',fontweight='bold')
    # plt.legend()
    p.axis([0.0,100,-0.25,1.25])
    i += 1
plt.show()
