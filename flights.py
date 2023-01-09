import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Bokningsdatum        0
# Avreseflygplats      0
# Ankomstflygplats     0
# Flygbolag            0
# Enkel_resa           0
# Antal_ben            0
# Flyg_vinst           0
# Extra_vinst          0
# Bagage_vinst         0
# Plats_vinst          0
# Utresedatum          0
# Hemresedatum         0
# Antal_bagage         0
# Antal_sittplatser    0
# Antal_vuxna          0
# Antal_barn           0

data = pd.read_csv('bokningsdata_viktor.csv')
data = data.dropna()
np_data = data.to_numpy()

print('number of categories')
cats = ['Avreseflygplats', 'Ankomstflygplats', 'Flygbolag']
for cat in cats:
    print(cat, len(data[cat].unique()))
    print('------')

# split data in ~2500 binary values
print('making pointers')
pointer = {}
def fun(x):
    n = 0
    while n < x:
        yield n
        n += 1
counter = fun(100000)
for i, elem in enumerate(np_data):
    #x
    for j in range(len(elem)):
        if j == 10 or j == 11:
            try:
                dt = datetime.strptime(elem[j], '%Y-%m-%d %H:%M:%S').date()
                wd = dt.weekday()
                pointer['d'+str(j+wd)] = pointer.get('d'+str(j+wd),None) if pointer.get('d'+str(j+wd),None) is not None else next(counter)
                m = dt.month
                pointer['m'+str(j+m)] = pointer.get('m'+str(j+m),None) if pointer.get('m'+str(j+m),None) is not None else next(counter)
                h = int(np_data[2][11][11:13])
                # hn, hm.. = hour night, hour morning, hour afternoon, hour evening
                if h < 6:
                    pointer['hn'] = pointer.get('hn',None) if pointer.get('hn',None) is not None else next(counter)
                elif h < 12:
                    pointer['hm'] = pointer.get('hm',None) if pointer.get('hm',None) is not None else next(counter)
                elif h < 17:
                    pointer['ha'] = pointer.get('ha',None) if pointer.get('ha',None) is not None else next(counter)
                else:
                    pointer['he'] = pointer.get('he',None) if pointer.get('he',None) is not None else next(counter)
            except:
                pass
        if j == 1 or j == 2 or j == 3 or j == 4 or j == 5:
            pointer[str(j)+str(elem[j])] = pointer.get(str(j)+str(elem[j]),None) if pointer.get(str(j)+str(elem[j]),None) is not None else next(counter)
no_cats = 0
for elem in pointer:
    no_cats = pointer[elem]+1

x = []
y = []
# create an array with 0s for all features, iterate the data and put the index pointer points on for the data to 1
print('making data')
for i, elem in enumerate(np_data):
    add = True
    x_row = [0 for _ in range(no_cats)]
    for j in range(len(elem)):
        if j == 10 or j == 11:
            try:
                dt = datetime.strptime(elem[j], '%Y-%m-%d %H:%M:%S').date()
                wd = dt.weekday()
                x_row[pointer['d'+str(j+wd)]] = 1
                m = dt.month
                x_row[pointer['m'+str(j+m)]] = 1
                h = int(np_data[2][11][11:13])
                if h < 6:
                    x_row[pointer['hn']] = 1
                elif h < 12:
                    x_row[pointer['hm']] = 1
                elif h < 17:
                    x_row[pointer['ha']] = 1
                else:
                    x_row[pointer['he']] = 1
            except:
                add = False
        if j == 1 or j == 2 or j == 3 or j == 4 or j == 5:
            x_row[pointer[str(j)+str(elem[j])]]
    if add:
        y.append(1 if elem[6]+elem[7]+elem[8]+elem[9]>0 else 0)
        x.append(np.array(x_row))
x_train, y_train, x_test, y_test = np.array(x[:-30000]), np.array(y[:-30000]), np.array(x[-30000:]), np.array(y[-30000:])

# Features not correlating well to target should likely be removed before this (atleast for logistic and forest).
# I would calculate correlation coef to determine that. 

print('logistic model')
clf = LogisticRegression(random_state=0).fit(x_train, y_train)
print(clf.score(x_test,y_test))

print('forest model')
forest_reg = RandomForestRegressor(n_jobs=-1, n_estimators=300)
forest_reg.fit(x_train,y_train)
print(confusion_matrix(y_test,(forest_reg.predict(x_test)).round()))

print('Dense NN')
model = Sequential()
model.add(Dense(60, input_shape=(len(x_train[0]),), activation='sigmoid'))
model.add(Dense(1, activation='sigmoid', kernel_regularizer = regularizers.l2(0.01)))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train,epochs=10, batch_size=64, validation_split=0.05)
loss, accuracy = model.evaluate(x_test,y_test)
print(accuracy)

# LSTM - maybe?

# should be plots for all categories here, find if there are any reasonalbe outliers profit could be saved on.
print('avg for company')
company_avg = {}
for i in range(len(np_data)):
    company_avg[np_data[i][3]] = company_avg.get(np_data[i][3],np.array([0,0])) + np.array([1,np_data[i][6]+np_data[i][7]+np_data[i][8]+np_data[i][9]])
y = []
z = []
n = []
for elem in company_avg:
    # print(f"{elem} = {company_avg[elem]}")
    y.append(company_avg[elem][1]/company_avg[elem][0])
    z.append(company_avg[elem][0])
    n.append(elem)

fig, ax = plt.subplots()
ax.scatter(z, y)

for i, txt in enumerate(n):
    ax.annotate(txt, (z[i], y[i]))
plt.xlabel("Number of times company was used")
plt.ylabel("Average profit for company")
plt.show()