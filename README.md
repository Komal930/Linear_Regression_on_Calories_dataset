# Linear_Regression_on_Calories_dataset
#importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns
import scipy
import pymc3 as pm
# reading files
exercise = pd.read_csv('exercise.csv')
calories = pd.read_csv('calories.csv')
df = pd.merge(exercise, calories, on = 'User_ID')
df = df[df['Calories'] < 300]
df = df.reset_index()
df['Intercept'] = 1
df.head()
#visuallizing data
plt.figure(figsize=(8, 8))
plt.plot(df['Duration'], df['Calories'], 'bo');
plt.xlabel('Duration (min)', size = 18); plt.ylabel('Calories', size = 18); 
plt.title('Calories burned vs Duration of Exercise', size = 20);
#defining variables (input and output)
X = df.loc[:, ['Intercept', 'Duration']]
y = df.loc[:, 'Calories']
#importing linearregression model
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(np.array(X.Duration).reshape(-1,1),y)
print('Intercept from library:', linear_regression.intercept_)
print('Slope from library:', linear_regression.coef_[0])
