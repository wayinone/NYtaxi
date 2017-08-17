# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 22:31:21 2017

@author: Wayne
"""
from importlib import reload
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import holidays
import AuxFun

import folium # goelogical map
import time
#%%
data = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv', parse_dates=['pickup_datetime'])# `parse_dates` will recognize the column is date time
test = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv', parse_dates=['pickup_datetime'])
data['store_and_fwd_flag'] = 1 * (data.store_and_fwd_flag.values == 'Y')
test['store_and_fwd_flag'] = 1 * (test.store_and_fwd_flag.values == 'Y')

#%%
outliers = pd.read_csv('../features/outliers.csv')

time_data       = pd.read_csv('../features/time_data.csv')
weather_data    = pd.read_csv('../features/weather_data.csv')
osrm_data       = pd.read_csv('../features/osrm_data.csv')
Other_dist_data = pd.read_csv('../features/Other_dist_data.csv')
#kmean20_data    = pd.read_csv('../features/kmean20_data.csv')
kmean10_data    = pd.read_csv('../features/kmean10_data.csv')

time_test       = pd.read_csv('../features/time_test.csv')
weather_test    = pd.read_csv('../features/weather_test.csv')
osrm_test       = pd.read_csv('../features/osrm_test.csv')
Other_dist_test = pd.read_csv('../features/Other_dist_test.csv')
#kmean20_test    = pd.read_csv('../features/kmean20_test.csv')
kmean10_test= pd.read_csv('../features/kmean10_test.csv')
#%%
train_loc = [None]*2;test_loc=[None]*2

kmean_data= pd.get_dummies(kmean10_data.pickup_dropoff_loc,prefix='loc', prefix_sep='_')    
kmean_test= pd.get_dummies(kmean10_test.pickup_dropoff_loc,prefix='loc', prefix_sep='_')    
#%% Kmeans 
#coords = np.vstack((data[['pickup_latitude', 'pickup_longitude']].values,
#                    data[['dropoff_latitude', 'dropoff_longitude']].values,
#                    test[['pickup_latitude', 'pickup_longitude']].values,
#                    test[['dropoff_latitude', 'dropoff_longitude']].values))
#sample_ind = np.random.permutation(len(coords))[:500000]
#kmeans = MiniBatchKMeans(n_clusters=20, batch_size=10000).fit(coords[sample_ind])
#
#for df in (data,test):
#    df.loc[:, 'pickup_loc'] = kmeans.predict(df[['pickup_latitude', 'pickup_longitude']])
#    df.loc[:, 'dropoff_loc'] = kmeans.predict(df[['dropoff_latitude', 'dropoff_longitude']])
#    
#N=50000
#plt.scatter(data.pickup_longitude[:N], data.pickup_latitude[:N], s=10, lw=0,
#           c=data.pickup_loc[:N].values, alpha=0.8,cmap='Set3')
#plt.xlim([-74.05,-73.75]);plt.ylim([40.6,40.9])
#plt.axes().set_aspect('equal')
#%% dummy for kmeans data
#train_loc = [None]*2;test_loc=[None]*2
#for i,loc in enumerate(['pickup_loc','dropoff_loc']):
#    train_loc[i]= pd.get_dummies(kmean20_data[loc], prefix=loc, prefix_sep='_')    
#    test_loc[i] = pd.get_dummies(kmean20_test[loc], prefix=loc, prefix_sep='_')
#kmean20_data = pd.concat(train_loc,axis=1)
#kmean20_test  = pd.concat(test_loc,axis=1)
#%%
mydf = data[['vendor_id','passenger_count','pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude','store_and_fwd_flag']]

testdf = test[['vendor_id','passenger_count','pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude','store_and_fwd_flag']]
mydf  = pd.concat([mydf  ,time_data,weather_data,osrm_data,Other_dist_data,kmean_data],axis=1)
testdf= pd.concat([testdf,time_test,weather_test,osrm_test,Other_dist_test,kmean_test],axis=1)
#%%
if np.all(mydf.keys()==testdf.keys()):
    print('Good! The keys of training feature is identical to those of test feature.')
    print('They both have %d features, as follows:'%len(mydf.keys()))
    print(list(mydf.keys()))
else:
    print('Oops, something is wrong, keys in training and testing are not matching')

#%% Generating map
map_1 = folium.Map(location=[40.767937,-73.982155 ],tiles='OpenStreetMap',
 zoom_start=12)

#tile: 'OpenStreetMap','Stamen Terrain','Mapbox Bright','Mapbox Control room'

for each in data.ix[outlier_id,:].iterrows():
    folium.CircleMarker([each[1]['pickup_latitude'],each[1]['pickup_longitude']],
                        radius=3,
                        color='red',
                        popup=str(each[1]['trip_duration']),
                        fill_color='#FD8A6C'
                        ).add_to(map_1)
map_1.save('map.html')
    

import googlemaps
#%%
from googlemaps import googlemaps as gmaps
#%% orig = (40.767937, -73.982155), destiney = (40.765602,-73.964630)
orig_lat = 40.767937
orig_lng = -73.982155
dest_lat = 40.765602
dest_lng = -73.964630
#api_key = 'AIzaSyAp-IgkbHn6ZwbZAZ0rmGc2FKxgPNHDrUQ'
#ans = gmaps.distance_matrix(client=api_key,origins=(orig_lat, orig_lng), destinations=(dest_lat, dest_lng))
#%%
import urllib, json, time
def google(lato, lono, latd, lond):
    url = """http://maps.googleapis.com/maps/api/distancematrix/json?origins=%s,%s"""%(lato, lono)+  \
    """&destinations=%s,%s&mode=driving&language=en-EN&sensor=false"""% (latd, lond)
    response = urllib.request.urlopen(url)
#    time.sleep(1)

    obj = json.load(response)
    try:
        minutes =   obj['rows'][0]['elements'][0]['duration']['value']/60
        miles = (obj['rows'][0]['elements'][0]['distance']['value']/1000) #kilometers
        return minutes, miles
    except IndexError:
        #something went wrong, the result was not found
        print (url)
        #return the error code
        return obj['Status'], obj['Status']

#%%
time,dist = google(orig_lat,orig_lng,dest_lat,dest_lng)
#The result is 7.95 min, distance 2.014km
#%%
fastrout = pd.read_csv('oscarleo_data/fastest_routes_train_part_1.csv')
#%%
fastrout.head()
#%%
a = fastrout.step_direction
b =  list(map(lambda x:x.count('right')-x.count('slight right'),a))
#%%
mydf = pd.read_csv('mydf.csv')
#%%
weather_cp = pd.read_csv('input/weather_data/weather_data_nyc_centralpark_2016.csv',parse_dates=['date'])

#%%
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
# setup Lambert Conformal basemap.
ax = Basemap(projection='cyl', 
            resolution = 'h', 
            llcrnrlon=-74.05,llcrnrlat=40.6,urcrnrlon=-73.75,urcrnrlat=40.95)
ax.drawcoastlines()
# draw a boundary around the map, fill the background.
# this background will end up being the ocean color, since
# the continents will be drawn on top.
ax.drawmapboundary(fill_color='aqua')
# fill continents, set lake color same as ocean color.
ax.fillcontinents(color='coral',lake_color='aqua')
plt.show()