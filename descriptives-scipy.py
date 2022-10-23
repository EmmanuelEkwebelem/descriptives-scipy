import pandas as pd
import numpy as np
import statsmodels
import seaborn
import scipy

from pandas.plotting import scatter_matrix
from scipy import stats
from statsmodels.formula.api import ols
from matplotlib import pyplot as plt

BrainData = pd.read_csv('C:/Users/emman/Desktop/HHA 507/descriptives-scipy/Data/brain_size.csv', sep=';', na_values=".")





### Exercise 1 ###

### Part A
BrainData.VIQ.mean()
### Part B
BrainData.Gender.value_counts()
### Part C 
BrainData_MRI = np.log(BrainData.MRI_Count.mean())





### Exercise 2 ###

pd.plotting.scatter_matrix(BrainData[['PIQ', 'VIQ', 'FSIQ']],  c=(BrainData['Gender'] == 'Female'))
pd.plotting.scatter_matrix(BrainData[['PIQ', 'VIQ', 'FSIQ']],  c=(BrainData['Gender'] == 'Male'))





### Exercise 3 ###

### Part A
female_weight = BrainData[BrainData['Gender'] == 'Female']['Weight']
male_weight = BrainData[BrainData['Gender'] == 'Male']['Weight']
stats.ttest_ind(female_weight, male_weight)   

### Part B
female_viq = BrainData[BrainData['Gender'] == 'Female']['VIQ']
male_viq = BrainData[BrainData['Gender'] == 'Male']['VIQ']
scipy.stats.mannwhitneyu(female_viq, male_viq)  





### Exercise 4 ###

x = np.linspace(-5, 5, 20)
np.random.seed(1)
y = -5 + 3*x + 4 * np.random.normal(size=x.shape)
data = pd.DataFrame({'x': x, 'y': y})
model = ols("y ~ x", data).fit()
print(model.summary())  





### Exercise 5 ###

model = ols('VIQ ~ C(Gender)', BrainData).fit()
model = ols('VIQ ~ Gender + MRI_Count + Weight + Height', BrainData).fit() 
print(model.f_test([0, 1, -1, 0, 0]))  