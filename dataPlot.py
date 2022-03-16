import matplotlib.pyplot as plt
import pandas as pd

data_train = pd.read_csv("data/train.csv")

fig = plt.figure()
fig.set(alpha=0.2)

ax1 = fig.add_subplot(141)
data_train.loc[(data_train['Sex'] == 'female') & (data_train['Pclass'] != 3), 'Survived'].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
ax1.set_xticklabels(['Survived', 'Dead'], rotation=0)
ax1.legend(['Femal/Highclass'], loc='best')

ax2=fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
ax2.set_xticklabels([u"Dead", u"Survived"], rotation=0)
plt.legend([u"Female/Low"], loc='best')

ax3=fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
ax3.set_xticklabels([u"Dead", u"Survived"], rotation=0)
plt.legend([u"Male/High"], loc='best')

ax4=fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
ax4.set_xticklabels([u"Dead", u"Survived"], rotation=0)
plt.legend([u"Male/Low"], loc='best')

# plt.subplot2grid((2, 3), (0, 0))
# data_train.Survived.value_counts().plot(kind='bar')
# plt.title(u"Survived(1) or Not(0)", fontsize='xx-small', fontweight='light')
# plt.ylabel(u"number", fontsize='xx-small', fontweight='light')
#
# plt.subplot2grid((2, 3), (0, 1))
# data_train.Pclass.value_counts().plot(kind="bar")
# plt.ylabel("number", fontsize='xx-small', fontweight='light')
# plt.title("Distribution of passenger's class", fontsize='xx-small', fontweight='light')
#
# plt.subplot2grid((2, 3), (0, 2))
# plt.scatter(data_train.Survived, data_train.Age)
# plt.ylabel("Age", fontsize='xx-small', fontweight='light')
# plt.grid(visible=True, which='major', axis='both')
# plt.title("Survived or not depends on age", fontsize='xx-small', fontweight='light')
#
# plt.subplot2grid((2, 3), (1, 0), colspan=2)
# data_train.Age[data_train.Pclass == 1].plot(kind='kde')
# data_train.Age[data_train.Pclass == 2].plot(kind='kde')
# data_train.Age[data_train.Pclass == 3].plot(kind='kde')
# plt.xlabel("Age", fontsize='xx-small', fontweight='light')
# plt.ylabel("Density", fontsize='xx-small', fontweight='light')
# plt.title("Distribution for the age of passengers", fontsize='xx-small', fontweight='light')
# plt.legend(('First Class', 'Second Class', 'Third Class'), loc='best')
#
# plt.subplot2grid((2, 3), (1, 2))
# data_train.Embarked.value_counts().plot(kind='bar')
# plt.title('Number of each port', fontsize='xx-small', fontweight='light')
# plt.ylabel('Number of People', fontsize='xx-small', fontweight='light')
#
# Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
# df = pd.DataFrame({"Survived": Survived_1, "Not-survived": Survived_0})
# df.plot(kind='bar', stacked=True)
# plt.title("Survived or not by class")
# plt.xlabel("Class")
# plt.ylabel("Number")
#
# fig = plt.figure()
# fig.set(alpha=0.2)
#
# Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
# Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
# df = pd.DataFrame({'Male': Survived_m, 'Female': Survived_f})
# df.plot(kind='bar', stacked=True)
# plt.title(u"按性别看获救情况")
# plt.xlabel(u"性别")
# plt.ylabel(u"人数")
plt.show()

fig=plt.figure()
fig.set(alpha=0.2)
Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'Survived':Survived_1, u'Dead':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"If survived depends on port")
plt.xlabel(u"Port")
plt.ylabel(u"Number")

plt.show()
