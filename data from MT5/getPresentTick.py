# -*- coding: utf-8 -*-

import MetaTrader5 as mt5
import pandas as pd


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
rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_D1, 0, 1)
 
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
print("\nDisplay dataframe with data of the current day:")
print(rates_frame) 

#Show collums of those 2 dClose + dTime
print("\n\n.CLose price of the current day:")
print(rates_frame[['time', 'close']])


#print("CLose hiện tại:\n ",dClose)