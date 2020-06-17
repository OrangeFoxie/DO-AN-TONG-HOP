# -*- coding: utf-8 -*-

#from datetime import datetime
import MetaTrader5 as mt5
import pandas as pd
#import plotly.graph_objects as go
import matplotlib.pyplot as plt
import math
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
import numpy as np
from pandas import Series
from numpy.random import randn

# display data on the MetaTrader 5 package
print("MetaTrader5 package author: ",mt5.__author__)
print("MetaTrader5 package version: ",mt5.__version__)


# import the 'pandas' module for displaying data obtained in the tabular form
pd.set_option('display.max_columns', 500) # number of columns to be displayed
pd.set_option('display.width', 1500)      # max table width to display
 
# establish connection to MetaTrader 5 terminal
if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()
 
# get 10000 GBPUSD D1 bars from the last 100 day
rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_D1, 1, 5000)
# Thấy dòng trên chứ !? Bạn nhìn thấy số 1 không, cạnh 5000 ấy. Ừm đúng r đó, nó đại diện cho số ngày trong MetaTrader5 đó.
# số 0 là hiện tại, số 1 là hôm qua, số 2 là ngày kia và cứ thế đếm ngc lại nha
# Cái Dự báo nó chạy cho 1 ngày tiếp theo của ngày hiện tại mà thớt chọn đó :3
 
# shut down connection to the MetaTrader 5 terminal
mt5.shutdown()
# display each element of obtained data in a new line
#print("Display obtained data 'as is'")
#for rate in rates:
#    print(rate)
 
# create DataFrame out of the obtained data
rates_frame = pd.DataFrame(rates)
# convert time in seconds into the datetime format
rates_frame['time']=pd.to_datetime(rates_frame['time'], unit='s')
 
# display data
print("\nDisplay dataframe with data")
print(rates_frame) 

#get close price and time
dClose = rates_frame.filter(['close'])
dTime = rates_frame.filter(['time'])
#Show collums of those 2 dClose + dTime
print(rates_frame[['time', 'close']])


#Visualize the closing price history in chart
plt.figure(figsize=(20,8))
plt.title('Close Price History from MT5')
plt.plot(dTime,dClose) #(Y,X)
plt.xlabel('Date',fontsize=10)
plt.ylabel('Close Price USD ($)',fontsize=10)
plt.grid()
plt.show()

# EVERYTHING IS DONE WITH GETTING DATA FROM MT5
# WE'RE GONNA USING THE CLOSE PRCIE TO TRAIN THEN PREDICT THE PRICE



#Converting the dataframe to a numpy array
dataset = dClose.values
#Get /Compute the number of rows to train the model on
training_data_len = math.ceil( len(dataset) *.8) 

#Scale the all of the data to be values between 0 and 1 
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data = scaler.fit_transform(dataset)



#Create the scaled training data set 
train_data = scaled_data[0:training_data_len  , : ]
#Split the data into x_train and y_train data sets
x_train=[]
y_train = []
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])




#Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)


#Reshape the data into the shape accepted by the LSTM
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


#Build the LSTM network model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


#Train the model
model.fit(x_train, y_train, batch_size=10, epochs=1)


#EROR FROM HERE
#Test data set
test_data = scaled_data[training_data_len - 60: , : ]
#Create the x_test and y_test data sets
x_test = []
y_test =  dataset[training_data_len : , : ] #Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])


#Convert x_test to a numpy array 
x_test = np.array(x_test)




#Reshape the data into the shape accepted by the LSTM
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))



#Getting the models predicted price values
predictions = model.predict(x_test) 
predictions = scaler.inverse_transform(predictions)#Undo scaling



#Calculate/Get the value of RMSE
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse


#Plot/Create the data for the graph

train = dClose[:training_data_len]
valid = dClose[training_data_len:]
valid['Predictions'] = np.squeeze(predictions)
#valid['Predictions'] = predictions

"""
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(dTime,train[['close']])
#plt.plot(dTime,dClose) 
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
"""

print("DATA after TRAINING")
valid


#SHOW THE VALUE OF PREDICTION
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date or DATA', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(valid)
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()






# THIS IS THE PREDICTION, IN WHICH THIS DUDE DO HIS JOB




#Get the quote
apple_quote = rates_frame
apple_quote
#Create a new dataframe
new_df = apple_quote.filter(['close'])
new_df
#Get the last 60 day closing price 
last_60_days = new_df[-60:].values
#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#Create an empty list
X_test = []
#Append the past 60 days
X_test.append(last_60_days_scaled)
#Convert the X_test data set to a numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Get the predicted scaled price
pred_price = model.predict(X_test)
#undo the scaling 
pred_price = scaler.inverse_transform(pred_price)
print("Dự báo cho ngày tiếp theo: ",pred_price)

















































