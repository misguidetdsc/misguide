#######################################################################
###################### Importing libraries ############################
#######################################################################

import numpy as np
import pandas as pd
import re
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


def distribute_loads(base_bus_loads, current_total_load):
    
    current_bus_loads = []
    base_total_load = sum(base_bus_loads)
    
    
    for i in range(len(base_bus_loads)):
         current_bus_loads.append( (base_bus_loads[i] / base_total_load) * current_total_load )   

    
    return current_bus_loads


def interpolate_loads(hourly_loads, degree):
    
    num_hours = len(hourly_loads)
    
    X = np.array([i * 3600 for i in range(num_hours)]).reshape(-1,1)
    y = hourly_loads.copy()        
     
    poly = PolynomialFeatures(degree = degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    
    
    pred_x = np.array(range(10800)).reshape(-1, 1)
    X_poly = poly.fit_transform(pred_x)
    per_second_loads = model.predict(X_poly)[:3600]
        
    return per_second_loads


def generate_clean_dataframe(raw_load_dataframe, base_bus_loads, imputation_look_back, base_mva):
    
    clean_load_dataframe = pd.DataFrame()
    
    years  = []
    months = []
    days   = []
    hours  = []
    times_of_day = []
    
    total_loads = []
    bus_loads = []
    
    for i in range(len(raw_load_dataframe) - imputation_look_back):
        print("Hour", i % 24, "Day", int(i / 24) + 1)
        date = re.split(' |-', str(raw_load_dataframe.iloc[i, 0]))
        current_total_load = raw_load_dataframe['Load (MW)'][i] / base_mva
        hourly_loads = (raw_load_dataframe['Load (MW)'][i: i + imputation_look_back] / base_mva).values
        
        per_second_loads = interpolate_loads(hourly_loads, degree = 3)
        
        for j in range(len(per_second_loads)):
            years.append(date[0])        
            months.append(date[1])        
            days.append(date[2])        
            hours.append(raw_load_dataframe['Hour'][i])
            times_of_day.append( (raw_load_dataframe['Hour'][i] - 1) * 3600 + j)
            
            total_loads.append(per_second_loads[j])
        
            current_bus_loads = distribute_loads(base_bus_loads, per_second_loads[j])
            bus_loads.append(current_bus_loads)
    
    bus_loads = np.array(bus_loads)
    
    clean_load_dataframe['Year']  = years
    clean_load_dataframe['Month'] = months
    clean_load_dataframe['Day']   = days
    clean_load_dataframe['Hour']  = hours
    clean_load_dataframe['Time of the Day (second)']  = times_of_day
    clean_load_dataframe['Total Load (PU)'] = total_loads
    
    for j in range(len(base_bus_loads)):
        clean_load_dataframe['Load Measurements in Bus' + str(j + 1) + " (PU)"] = bus_loads[:, j]

    return clean_load_dataframe


num_samples          = 724
imputation_look_back = 4
base_mva             = 100

raw_load_dataframe = pd.read_excel('data/Raw-Load-Dataset_39-Bus.xlsx').iloc[:num_samples,:]
base_parameters = pd.read_csv('data/Bus-Parameters_39-Bus.csv')

base_bus_loads = base_parameters['Load Active Power (PU)'].values

clean_load_dataframe = generate_clean_dataframe(raw_load_dataframe, base_bus_loads, imputation_look_back, base_mva)

plt.plot(clean_load_dataframe['Total Load (PU)'])
clean_load_dataframe.to_csv('data/Cleaned-load-Dataframe_30-Days.csv', index = False) 
