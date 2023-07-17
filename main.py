import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
source_file = pd.read_csv('AMZN.csv')
print(source_file.head()) #inspects thef first few rows of the source file

#Data Preprocessing
source_file.dropna(inplace=True) #Drops rows with missing values
source_file['Date'] = pd.to_datetime(source_file['Date'])
source_file.sort_values('Date', inplace=True)


#Augmented Dickey-Fuller to check if the values are stationary or not
target_column = 'Close'
adf_result = adfuller(source_file[target_column])

print('ADF Statistic:', adf_result[0]) #usually given a negative number, if a strong negative number then we reject the hypthesis that there is a unit root
print('p-value:', adf_result[1]) # if p-value is low, series is stationary
print('Critical Values:')
for key, value in adf_result[4].items():
    print('\t{}: {}'.format(key, value))

#Plotting the rolling statistics, visualization of stationary values
rolling_mean = source_file[target_column].rolling(window=12).mean()
rolling_std = source_file[target_column].rolling(window=12).std() #Constant rolling sd and mean means the data is stationary
plt.plot(source_file[target_column], label='Original')
plt.plot(rolling_mean, label='Rolling Mean')
plt.plot(rolling_std, label='Rolling Std')
plt.legend()
plt.xlabel('Date') #X-value
plt.ylabel('Closing Price') #y-value
plt.title('Rolling Statistics')
plt.show()
#Need to close plot in order for program to move on

#Training the data
train_size = int(len(source_file) * 0.8)
train_data = source_file.iloc[:train_size]
test_data = source_file.iloc[train_size:]

#Finding the p, d and q parameters for the ARIMA model manually by the user looking at the graphs
#p-parameter(autoregressive order), d(difference order), q(moving average order)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(source_file[target_column], ax=ax1, lags=40) #Significant spikes show the need for a q-parameter
plot_pacf(source_file[target_column], ax=ax2, lags=40) #Siginifcant spikes show the need for a p-parameter
plt.show()
#d paramter found by analyzing the difference level to make the data stationary

#automatically finding the p,d and q parameters(go to option)
model1 = auto_arima(source_file['Close'], seasonal=False, trace=True)
p, d, q = model1.order

#Creating model and fitting the data
model = ARIMA(train_data['Close'], order=(p, d, q))
model_fit = model.fit()

#Validating the performance by analyzing the mean squared error and mean average error
predictions = model_fit.predict(start=test_data.index[0], end=test_data.index[-1]) 
mse = ((predictions - test_data['Close']) ** 2).mean() #lower MSE value means better model
mae = (abs(predictions - test_data['Close'])).mean() # lower MAE value means better model
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)


#Fitting the model to the whole set and not just the test data
model_final = ARIMA(source_file['Close'], order=(p, d, q))
model_final_fit = model_final.fit()

#Forecasting the future prices
#num_forecast_periods is how many periods in the future you want to see
num_forecast_periods = 10
forecast = model_final_fit.predict(start=len(source_file), end=len(source_file) + num_forecast_periods)

# Visualize the predicted stock prices and compare them with the actual prices
plt.plot(source_file['Close'], label='Actual')
plt.plot(predictions, label='Predicted')
plt.plot(forecast, label='Forecast')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()