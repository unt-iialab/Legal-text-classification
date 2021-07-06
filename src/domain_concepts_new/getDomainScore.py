import pandas as pd
from sklearn import preprocessing
import operator

dataframe = pd.read_csv('/home/junhua/PycharmProjects/concept_normalization/domainconcepts/ccnu_data/importantConcepts_0705', sep=",")
print(dataframe.head(30))
print(dataframe.shape)
dataframe = dataframe.drop('leixing', axis=1)
print(dataframe.head())
column_header = list(dataframe.columns)
print(list(dataframe.columns))

print("------step1------")
x = dataframe.values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# df = pd.DataFrame(x_scaled)
print(dataframe.head())
# average_domain_score = df.mean().tolist()
# print(df.mean().tolist())

# res = dict(zip(column_header, average_domain_score))
# print(res)
# sort_res = dict( sorted(res.items(), key=operator.itemgetter(1),reverse=True))
# print(sort_res)

category_42 = dataframe.iloc[0].tolist()
category_20 = dataframe.iloc[1].tolist()
category_36 = dataframe.iloc[4].tolist()
category_16 = dataframe.iloc[6].tolist()
category_1 = dataframe.iloc[9].tolist()
category_56 = dataframe.iloc[28].tolist()
# print(category_42)

category_42_dict = dict(zip(column_header, category_42))
category_20_dict = dict(zip(column_header, category_20))
category_36_dict = dict(zip(column_header, category_36))
category_16_dict = dict(zip(column_header, category_16))
category_1_dict = dict(zip(column_header, category_1))
category_56_dict = dict(zip(column_header, category_56))

sort_res_42 = dict( sorted(category_42_dict.items(), key=operator.itemgetter(1),reverse=True))
sort_res_20 = dict( sorted(category_20_dict.items(), key=operator.itemgetter(1),reverse=True))
sort_res_36 = dict( sorted(category_36_dict.items(), key=operator.itemgetter(1),reverse=True))
sort_res_16 = dict( sorted(category_16_dict.items(), key=operator.itemgetter(1),reverse=True))
sort_res_1 = dict( sorted(category_1_dict.items(), key=operator.itemgetter(1),reverse=True))
sort_res_56 = dict( sorted(category_56_dict.items(), key=operator.itemgetter(1),reverse=True))

print(sort_res_1)
print(sort_res_16)
print(sort_res_20)
print(sort_res_36)
print(sort_res_42)
print(sort_res_56)