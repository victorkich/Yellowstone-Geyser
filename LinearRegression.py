import pandas as pd
from sklearn import linear_model
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split 

# Select the DataSet
df = pd.read_csv('GeyserUFSM.csv')

# Select the values for predict 
prever = pd.DataFrame({'Data':[200, 230, 245, 270]})

# Create LinearRegression function
regr = linear_model.LinearRegression()

# Split your data into X and y 
X = np.array(df['espera']).reshape(-1, 1) 
y = np.array(df['erupcao']).reshape(-1, 1) 

# Use the train_test_split for split your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30) 

# Apply the prediction model
regr.fit(X_train, y_train)
predicao = regr.predict(np.array(prever['Data']).reshape(-1, 1))

# Print score and predictive individual values
print(regr.score(X_test, y_test)) 
print(predicao)

# Plot
y_pred = regr.predict(X_test) 
plt.scatter(X_test, y_test, color ='b') 
plt.plot(X_test, y_pred, color ='k') 
plt.show() 