import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import Series, DataFrame

data_train = pd.read_csv("data/train.csv")

fig = plt.figure()
fig.set(alpha=0.2)

plt.subplot2grid((2, 3), (0, 0))
data_train.Survived.value_counts().plot(kind='bar')
plt.title(u"Survived(1) or Not(0)", fontsize='xx-small', fontweight='light')
plt.ylabel(u"number", fontsize='xx-small', fontweight='light')

plt.subplot2grid((2, 3), (0, 1))
data_train.Pclass.value_counts().plot(kind="bar")
print(data_train.Survived.value_counts())
plt.ylabel("number", fontsize='xx-small', fontweight='light')
plt.title("Distribution of passenger's class", fontsize='xx-small', fontweight='light')

plt.subplot2grid((2, 3), (0, 2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel("Age", fontsize='xx-small', fontweight='light')
plt.grid(visible=True, which='major', axis='both')
plt.title("Survived or not depends on age", fontsize='xx-small', fontweight='light')

plt.subplot2grid((2, 3), (1, 0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel("Age", fontsize='xx-small', fontweight='light')
plt.ylabel("Density", fontsize='xx-small', fontweight='light')
plt.title("Distribution for the age of passengers", fontsize='xx-small', fontweight='light')
plt.legend(('First Class', 'Second Class', 'Third Class'), loc='best')

plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title('Number of each port', fontsize='xx-small', fontweight='light')
plt.ylabel('Number of People', fontsize='xx-small', fontweight='light')

plt.show()
