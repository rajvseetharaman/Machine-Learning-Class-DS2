
# coding: utf-8

# ###### Problem Set 1, due January 10th at 5:30pm 
# 
# 
# ###Before You Start
# Make sure to at least take a basic tutorial in the IPython notebook, otherwise you'll be totally lost.  For this problem set, you should download INFX574-PS1.ipynb and the flights.zip dataset from Canvas. Create a local copy of the notebook and rename it LASTNAME_FIRSTNAME-PS1.ipynb. Then edit your renamed file directly in your browser by typing:
# ```
# ipython notebook <name_of_downloaded_file>
# ```
# 
# You should also make sure the following libraries load correctly (click on the box below and hit Ctrl-Enter)

# In[130]:


# #IPython is what you are using now to run the notebook
# import IPython
# print "IPython version:      %6.6s (need at least 1.0)" % IPython.__version__

# Numpy is a library for working with Arrays
import numpy as np
print("Numpy version:        %6.6s (need at least 1.7.1)" % np.__version__)

# SciPy implements many different numerical algorithms
import scipy as sp
print("SciPy version:        %6.6s (need at least 0.12.0)" % sp.__version__) 

# Pandas makes working with data tables easier
import pandas as pd
print("Pandas version:       %6.6s (need at least 0.11.0)" % pd.__version__) 

# Module for plotting
import matplotlib
print("Mapltolib version:    %6.6s (need at least 1.2.1)" % matplotlib.__version__) 

# SciKit Learn implements several Machine Learning algorithms
import sklearn
print("Scikit-Learn version: %6.6s (need at least 0.13.1)" % sklearn.__version__) 


# ##About the Problem Set: 
# This is the same problem set used by Emma Spiro in INFX573. The only difference is that instead of doing the problem set in R, you will use Python and the IPython notebook.
# 
# ##Instructions: 
# In this problem set you will perform a basic exploratory analysis on an example dataset, bringing to bear all of your new skills in data manipulation and visualization. You will be required to submit well commented python code, documenting all code used in this problem set, along with a write up answering all questions below. Use figures as appropriate to support your answers, and when required by the problem. 
# This data set uses the NYCFlights13 dataset. You can download the dataset from canvas.
# Selected questions ask you to answer in multiple ways. Make sure to provide different functions or ways for answering the same question. This will help you see that most data questions can be answered in different ways even with the same software language.

# In[131]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[132]:


#read flights data
flights_df= pd.read_csv('flights.csv')


# In[133]:


#summarize flights data
print(flights_df.shape)
print(flights_df.columns)
print(flights_df.dtypes)


# In[134]:


#print unique destinations
a = flights_df.dest.unique()
print(a)
#first 10 rows in data
flights_df.head(10)


# ##Some Tips
# 
# * This assignment involves extensive Data frame splitting and aggregation. You should look into the details of the methods groupby, transform, sum, count, mean etc
# * Many of the tasks in the assignment can be done either through the Pandas Data Frame or by converting the data frames to Series. Many of the methods in the numpy are applicable to Series only. When stuck, try to explore the type of object (Pandas Data Frame or Numpy Series) you are dealing with.

# ##Question 1
# Let’s explore flights from NYC to Seattle. Use the flights dataset to answer the following questions.
# 
# (a) How many flights were there from NYC airports to Seattle in 2013?

# In[135]:


#filter flights with destination as seattle
flights_df1=flights_df[ flights_df.dest=='SEA']
#find count of flights
flights_df1.shape[0]


# Ans. There were 3923 flights from NYC airports to Seattle in 2013

# (b) How many airlines fly from NYC to Seattle?

# In[136]:


#find unique carriers in the flights to seattle
carriersN_to_seattle = flights_df1.carrier.unique()
print(carriersN_to_seattle)


# Ans. 5 airlines fly from NYC to Seattle.

# (c) How many unique air planes fly from NYC to Seattle?

# In[137]:


#count of unique airplanes which fly to seattle
flights_df1['tailnum'].nunique()


# Ans. 935 unique air planes fly from NYC to Seattle.

# (d) What is the average arrival delay for flights from NC to Seattle?

# In[138]:


#find mean arrival delay of flights from nyc to seattle
flights_df1['arr_delay'].mean()


# Ans. The average arrival delay for flights from NYC to Seattle is -1.099 mins.

# (e) What proportion of flights to Seattle come from each NYC airport? Provide multiple ways of answering the question.

# In[139]:


#find count of flights to seattle which come from each origin NYC airport
df2=flights_df1.groupby(['origin'])['origin'].count()
#transformation function to convert count to proportion
prop_trans=lambda x:x*100/x.sum()
#use above function to fnd proportion of flights from each NYC airport to seattle
print(df2.transform(prop_trans))    

# find unique origins
nyc_origins = flights_df1['origin'].unique()
print(nyc_origins)

d = {'col1': nyc_origins, 'col2': nyc_origins}
print(d)

for i in range(0, len(nyc_origins)):
               print(nyc_origins[i])
#count flights coming from EWR
a = flights_df1[flights_df1.origin == 'EWR']['origin'].count()
#count flights coming from JFK
b = flights_df1[flights_df1.origin == 'JFK']['origin'].count()
#count flights coming from JFK
c = flights_df1[flights_df1.origin == 'LGA']['origin'].count()
total_flights=a+b+c
#compute proportion of flights coming from each NYC airport to seattle
prop_ewr=a*100/total_flights
prop_jfk=b*100/total_flights
prop_lga=c*100/total_flights
print('Proportion of flights from EWR: ',prop_ewr)
print('Proportion of flights from JFK: ',prop_jfk)
print('Proportion of flights from LGA: ',prop_lga)


# Ans.
# Proportion of flights from EWR is  46.67% and from JFK is 53.33%

# ## Question 2
# Flights are often delayed. Consider the following questions exploring delay patterns.
# 
# (a) Which date has the largest average departure delay? Which date has the largest average arrival delay?

# In[11]:


# find average departure delay for each day
df2=flights_df.groupby(['year','month','day'])['dep_delay'].agg(np.mean)
#find the day with largest departure delay
df2[df2.values == df2.values.max()]


# In[140]:


# find average arrival delay for each day
df2=flights_df.groupby(['year','month','day'])['arr_delay'].agg(np.mean)
#find the day with largest arrival delay
df2[df2.values == df2.values.max()]


# Ans. The day 03/08/2013 had both the largest average arrival and departure delays.

# (b) What was the worst day to fly out of NYC in 2013 if you dislike delayed flights?

# In[141]:


#filter all flights having a departure delay and find count of delayed flights for each day
df3=flights_df[flights_df.dep_delay>0].groupby(['year','month','day'])['dep_delay'].count()
#find the day with maximum delayed flights
df3[df3.values == df3.values.max()]


# Ans. 
# The metric that I am using to judge if a particular day is bad to fly is if that day has a high number of delayed flights.
# 23 December 2013 was the worst day to fly out of NYC according to me because it had the most number of flights (674) with departure delays.
# 
# If we consider the average departure delay as the metric for determining a bad day to fly, 8th March 2013 was a bad day to fly with an average departure delay of 83.53 mins.

# (c) Are there any seasonal patterns in departure delays for flights from NYC?

# In[142]:


get_ipython().run_line_magic('matplotlib', 'inline')
#find average departure delay by month
df4 = flights_df.groupby(['month'])['dep_delay'].agg(np.mean)
print(df4)
#plot avg departure delay by month
df4.plot()


# Ans. The data seems to show seasonal patterns in departure delays. The summer months of june july seem to have greater average departure delays in the year. The months of september october and november seem to show lower departure delays with sharp spike in departure delays in the month of December probably because a lot of people travel for the holidays resulting in greater flight delays. 

# (d) On average, how do departure delays vary over the course of a day?

# In[143]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', 'inline')
#find avg departure delay  by hour
print(flights_df.groupby(['hour'])['dep_delay'].agg(np.mean))
pylab.rcParams['figure.figsize'] = (14, 9)
#find avg departure delay  by hour,minute
df5 = flights_df.groupby(['hour','minute'])['dep_delay'].agg(np.mean)
#plot hourly variation in departure delay
df5.plot(x=['hour','minute'])


# Ans. Looking at the above data and the plot, we can see that the average departure delays are the high at 12 am and spike at around 3 am, followed by a very sharp dip in delays at 4 am. The departure delays then steadily increase throughout the day followed by sharp increases in delay at around 9 pm.

# ## Question 3
#     Which flight departing NYC in 2013 flew the fastest?

# In[144]:


#retain necessary variables
df6=flights_df[['carrier','flight','origin','dest','distance','air_time']]
#compute additional variable flight speed
df6['flight_speed']=df6.distance*60/df6.air_time
#find flight having max flight speed
df6[df6.flight_speed==df6.flight_speed.max()]


# Ans. Flight 216447 by Delta airlines from LGA to ATL flew the fastest with a flight speed of 703 miles/hr, covering a distance of 762 miles in a span of 65 mins.

# ## Question 4
# Which flights (i.e. carrier + flight + dest) happen every day? Where do they fly to?

# In[145]:


#find count of flights for each combination of carrier, flight and destination
df7=flights_df.groupby(['carrier','flight','dest'])['flight'].count()
#if the count is 365 for the above combination, it implies that that particular flight has flown every day
df8=df7[df7.values==365]
#print flights that have flown all year
print(df8)
#find the unique carriers who have flown the flights and the destinations to which flights have flown all year
print(df8.index.get_level_values('carrier').unique())
print(df8.index.get_level_values('dest').unique())


# Ans. The above flights happen every day. The destinations for these flights are 'SFO', 'LAX', 'SJU', 'MIA', 'CLT', 'BUR', 'FLL', 'SRQ', 'MCO', 'TPA','IAD', 'HNL', and 'LAS'

# ## Question 5
# Develop one research question you can address using the nycflights2013 dataset. Provide two visualizations to support your exploration of this question. Discuss what you find.
# 

# In[146]:


#compute hourly average departure delays for each origin airport in NYC i.e JFK,EWR,LGA
hourly_delay_origin=flights_df.groupby(['origin','hour'])['dep_delay'].agg(np.mean).reset_index()
jfk=hourly_delay_origin[hourly_delay_origin.origin=='JFK']
lga=hourly_delay_origin[hourly_delay_origin.origin=='LGA']
ewr=hourly_delay_origin[hourly_delay_origin.origin=='EWR']


# Plotting departure delay over the course of the day for JFK
plt.bar(jfk.hour, jfk.dep_delay, 0.70)
plt.xlabel('Hour of the Day')
plt.ylabel('Average Departure Delay')
plt.title('Average Departure Delay over the course of a day for JFK')
plt.show()

# Plotting departure delay over the course of the day for LGA
plt.bar(lga.hour, lga.dep_delay, 0.70)
plt.xlabel('Hour of the Day')
plt.ylabel('Average Departure Delay')
plt.title('Average Departure Delay over the course of a day for LGA')
plt.show()

# Plotting departure delay over the course of the day for EWR
plt.bar(ewr.hour, ewr.dep_delay, 0.70)
plt.xlabel('Hour of the Day')
plt.ylabel('Average Departure Delay')
plt.title('Average Departure Delay over the course of a day for EWR')
plt.show()


# Ans. My research question:
# 
# Is the hourly pattern of departure delays in a day different for the different airports in NYC i.e JFK, EWR, and LGA? 
# 
# Looking at the above plots, I see that the departure delay hourly pattern for each of the airports closely resembles that of the overall hourly delay patterns. One noticable difference I see between the airports is that for the LGA and EWR airport, after the sharp drop in departure delays, the departure delays remain low until late in the day when there is a sharp spike in the departure delays for both airports. For JFK, there is an upward spike too, but it is not as sharp as the other 2 airports in NYC. Another observation is that JFK has its highest avg departure delays at around 3 pm while the other 2 airports reach that peak at around 2 pm, followed by a sharp fall in delays. 
# 

# ## Question 6
# What weather conditions are associated with flight delays leaving NYC? Use graphics to explore.

# In[147]:



# Loading the weather.csv file
weather_df = pd.read_csv('weather.csv')  
# find hourly average departure delays
grouped_fl=flights_df.groupby(['year','month','day','hour'])['dep_delay'].agg(np.mean).reset_index()
#merge above departure delays data with hourly weather data using hour, day, month, year variables
df=pd.merge(grouped_fl,weather_df,on=['year','month','day','hour'])

# Plotting the graphs
matplotlib.style.use('ggplot')
rcParams['figure.figsize'] = 30, 30
fig = plt.figure()

# Plotting Temp Vs. Departure Delay graph
fig1 = fig.add_subplot(331)
fig1.scatter(df['temp'], df['dep_delay'])
xlabel('Avg Temp in F')
ylabel('Avg departure delay in mins')
title('Temp Vs. Departure Delay')

# Plotting Visibility Vs. Departure Delay graph
fig2 = fig.add_subplot(332)
fig2.scatter(df['visib'], df['dep_delay'])
xlabel('Average Visibility in miles')
ylabel('Avg departure delay in mins')
title('Visibility Vs. Departure Delay')

# Plotting Dew point Vs. Departure Delay graph
fig3 = fig.add_subplot(333)
fig3.scatter(df['dewp'],df['dep_delay'])
xlabel('dewpoint in F')
ylabel('Avg departure delay in mins')
title('Dewpoint Vs. Departure Delay')

# Plotting Precipitation Vs. Departure Delay graph
fig4 = fig.add_subplot(334)
fig4.scatter(df['precip'], df['dep_delay'])
xlabel('Avg precipitation in inches')
ylabel('Avg departure delay in mins')
title('Precipitation Vs. Departure Delay')

# Plotting Relative Humidity Vs. Departure Delay graph
fig5 = fig.add_subplot(335)
fig5.scatter(df['humid'], df['dep_delay'])
xlabel('Relative humidity')
ylabel('Avg departure delay in mins')
title('Relative Humidity Vs. Depature Delay')

# Plotting Pressure Vs. Departure Delay graph
fig6 = fig.add_subplot(336)
fig6.scatter(df['pressure'], df['dep_delay'])
xlabel('Sea level pressure in millibars')
ylabel('Avg departure delay in mins')
title('Pressure Vs. Depature Delay')


# Plotting Wind Speed Vs. Departure Delay graph
fig7 = fig.add_subplot(337)
fig7.scatter(df['wind_speed'], df['dep_delay'])
xlabel('Wind speed in mph')
ylabel('Avg departure delay in mins')
title('Wind Speed Vs. Depature Delay')


# Plotting Wind Gust Vs. Departure Delay graph
fig8 = fig.add_subplot(338)
fig8.scatter(df['wind_gust'], df['dep_delay'])
xlabel('Wind gust in mph')
ylabel('Avg departure delay in mins')
title('Wind Gust Vs. Depature Delay')

# Plotting Wind Direction Vs. Departure Delay graph
fig9 = fig.add_subplot(339)
fig9.scatter(df['wind_dir'], df['dep_delay'])
xlabel('Wind direction in degrees')
ylabel('Avg departure delay in mins')
title('Wind Direction Vs. Depature Delay')

plt.show()

#finding correlation between weather variables and departure delays

print('Correlation between weather variables and departure delays')
print('Temperature and departure delay')
print(df['temp'].corr(df['dep_delay']))
print('Visibility and departure delay')
print(df['visib'].corr(df['dep_delay']))
print('Dew point and departure delay')
print(df['dewp'].corr(df['dep_delay']))
print('Precipitation and departure delay')
print(df['precip'].corr(df['dep_delay']))
print('Humidity and departure delay')
print(df['humid'].corr(df['dep_delay']))
print('Pressure and departure delay')
print(df['pressure'].corr(df['dep_delay']))
print('Wind speed and departure delay')
print(df['wind_speed'].corr(df['dep_delay']))
print('Wind gust and departure delay')
print(df['wind_gust'].corr(df['dep_delay']))
print('Wind direction and departure delay')
print(df['wind_dir'].corr(df['dep_delay']))


# Looking at the graphs, I see that weather conditions do affect the departure delays at the NYC airports. 
# 
# a) Temperature seems to impact the departure delay. With an increase in temperature, there is an increase in departure delay.
# b) Visibility also seems to be impacting the departure delays. This is obvious because low visibility conditions make it difficult for flights to take off.
# c) Dewpoint also seems to be impact the departure delay. With increase in dewpoint, there is an increase in departure delays. 
# d) Precipitation also seems to be impact departure delay. Rise in precipitation increases the departure delay. Rainy conditions make it difficult for flights to take off.
# d) Rise in relative Humidity also seems to increase the departure delay.
# e) Increase in pressure seems to decrease flight delays.
# Wind direction, speed, and gust do not seem to impact delays as much which can been seen by low correlation coefficients.
