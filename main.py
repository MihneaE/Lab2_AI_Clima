# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

data = pd.read_csv("GLB.Ts+dSST (3).csv")

#ANALIZA EXPLORATORIE A DATELOR FOLOSITA CA INPUT PENTRU PROIECT

#Description OF Data
print(data.describe())

#Handling missing data

#drop NUll or missing values
data_clean = data.dropna()
data_clean.to_csv('clean_data.csv', index = False)

#fill Missing values
#umple toate valorile lipsa cu 0
data_filled = data.fillna(value=0)
data_filled.to_csv('data_filled.csv', index = False)

#BoxPlot
# Convertim coloana la tipul float
data['Global Temperature Anomalies (deg C) AIRS v6 vs. 2007-2016'] = pd.to_numeric(data['Global Temperature Anomalies (deg C) AIRS v6 vs. 2007-2016'], errors='coerce')

plt.boxplot(data['Global Temperature Anomalies (deg C) AIRS v6 vs. 2007-2016'])
#plt.show()

# calculăm z-score
z_scores = zscore(data['Global Temperature Anomalies (deg C) AIRS v6 vs. 2007-2016'])
# creăm o nouă coloană pentru z-score
data['Z_scores'] = z_scores
# afișăm rândurile unde z-score este mai mare ca 3 sau mai mic ca -3
outliers = data[(data['Z_scores'] > 3) | (data['Z_scores'] < -3)]
print(outliers)

# Calcularea IQR
Q1 = data['Global Temperature Anomalies (deg C) AIRS v6 vs. 2007-2016'].quantile(0.25)
Q3 = data['Global Temperature Anomalies (deg C) AIRS v6 vs. 2007-2016'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Detectarea outlierilor
outliers2 = data[(data['Global Temperature Anomalies (deg C) AIRS v6 vs. 2007-2016'] < lower_bound) | (data['Global Temperature Anomalies (deg C) AIRS v6 vs. 2007-2016'] > upper_bound)]
print(outliers2)

#HISTORIGRAMA
# asigurați-vă că datele sunt curățate și pregătite înainte de a genera histograma
cleaned_data = data['Global Temperature Anomalies (deg C) AIRS v6 vs. 2007-2016'].dropna()

# generarea histograma
plt.hist(cleaned_data, bins=30, edgecolor='black')
plt.title('Histogram of Global Temperature Anomalies')
plt.xlabel('Temperature Anomalies (deg C)')
plt.ylabel('Frequency')
#plt.show()

#HEATMAP
# calculăm matricea de corelație
corr = data.corr()
# generăm heatmap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap='RdBu')
#plt.show()

#Standardization
# cream un scaler
scaler = MinMaxScaler()
# fit si transform pe date
data['Global Temperature Anomalies (deg C) AIRS v6 vs. 2007-2016'] = scaler.fit_transform(data['Global Temperature Anomalies (deg C) AIRS v6 vs. 2007-2016'].values.reshape(-1,1))

# creăm un scaler
scaler = RobustScaler()
# fit și transform pe date
data['Global Temperature Anomalies (deg C) AIRS v6 vs. 2007-2016'] = scaler.fit_transform(data['Global Temperature Anomalies (deg C) AIRS v6 vs. 2007-2016'].values.reshape(-1,1))

#Discretization
data['Temperature Discrete'] = pd.cut(data['Global Temperature Anomalies (deg C) AIRS v6 vs. 2007-2016'], bins=10, labels=False)


#Encoding categorical features
labelencoder = LabelEncoder()
data['Temperature Discrete'] = labelencoder.fit_transform(data['Temperature Discrete'])
data['Global Temperature Anomalies (deg C) AIRS v6 vs. 2007-2016'] = labelencoder.fit_transform(data['Global Temperature Anomalies (deg C) AIRS v6 vs. 2007-2016'])

