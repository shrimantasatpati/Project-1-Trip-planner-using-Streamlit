import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_folium import folium_static
sns.set()
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import folium
from folium.features import DivIcon
import streamlit as st

# Define a function to load the data
@st.cache_data
def load_data():
    df = pd.read_csv("Unique_places.csv")
    return df

# Load the data
df = load_data()

# Create a DataFrame with just the relevant columns
l2 = df[["Place", "latitude", "longitude"]]

# Perform K-means clustering
places_lat_long = l2[['latitude', 'longitude']].values.tolist()
places_lat_long = np.array(places_lat_long)
kmeans = KMeans(n_clusters=40, random_state=0).fit(places_lat_long)
group = list(kmeans.labels_)
l2['cluster'] = pd.Series(group, index=l2.index)

# Define a function to get the cluster number for a given city
@st.cache_data
def get_cluster(city):
    cluster = l2.loc[df['Place'] == city]['cluster'].values[0]
    return cluster

# Define a function to get an itinerary for a given city and number of days
@st.cache(suppress_st_warning=True)
def get_itinerary(city, numdays):
    cluster = get_cluster(city)
    cities_in_cluster = pd.DataFrame(l2.loc[l2['cluster'] == cluster])
    places_lat_long = cities_in_cluster[['latitude','longitude']].values.tolist()
    places_lat_long = np.array(places_lat_long)
    kmeans = KMeans(n_clusters=int(numdays), random_state=0).fit(places_lat_long)
    group = list(kmeans.labels_)
    cities_in_cluster['subCluster'] = pd.Series(group, index=cities_in_cluster.index)
    cities_with_days = cities_in_cluster[["Place", "latitude", "longitude", "subCluster"]]
    mean_lat_long_by_group = cities_with_days.groupby('subCluster')[['latitude', 'longitude']].mean()
    distance_matrix = cdist(mean_lat_long_by_group.values, mean_lat_long_by_group.values)
    df_distance_matrix = pd.DataFrame(distance_matrix)
    starting_point = 2
    cur_index = starting_point
    seq = [cur_index]
    while len(seq) < len(list(df_distance_matrix.keys())):
        nearest_clusters = list(df_distance_matrix[cur_index].sort_values().index)
        for cluster_id in nearest_clusters:
            if cluster_id != cur_index and cluster_id not in seq:
                seq.append(cluster_id)
                cur_index = cluster_id
                break
    replace_group_to_day = {}
    for i in range(0, len(seq)):
        replace_group_to_day[seq[i]] = i
    cities_with_days['days'] = cities_with_days['subCluster'].apply(lambda x: replace_group_to_day[x])

    # Return the itinerary as a DataFrame
    return cities_with_days

# Define the Streamlit app
def app():
    st.title("Trip Planner")

    # Create a dropdown to select the city
    city = st.selectbox("Select a city", df["Place"])

    # Create a slider to select the number of days
    numdays = st.slider("Select the number of days", 1, 10, 5)

    # Display the itinerary for the selected city and number of days
    itinerary = get_itinerary(city, numdays)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    prev_location = None
    # Display a map of the places in the itinerary using Folium
    map_center = itinerary[['latitude', 'longitude']].mean().values.tolist()
    map = folium.Map(location=map_center, zoom_start=5)
    for index, row in itinerary.iterrows():
        location = (row['latitude'], row['longitude'])
        day_number = row['days']
        color = colors[day_number % len(colors)]
        icon_text = str(row['days']+1)
        icon_style = f"background-color: {color}; border-radius: 50%; width: 30px; height: 30px; display: flex; justify-content: center; align-items: center;"
        icon = DivIcon(icon_size=(30, 30), icon_anchor=(15, 15), html=f"<div style='{icon_style}'>{icon_text}</div>")
        popup = f"{row['Place']}"
        marker = folium.Marker(location=location, icon=icon, popup=popup)

        # Choose a different color for the marker based on the day number
        if row['days'] == 1:
            marker.add_to(map).icon = folium.Icon(color='red')
        elif row['days'] == itinerary['days'].max():
            marker.add_to(map).icon = folium.Icon(color='green')
        else:
            marker.add_to(map).icon = folium.Icon(color='blue')

        # Connect this location to the previous location with a line, if applicable
        if prev_location is not None:
            folium.PolyLine(locations=[prev_location, location], color=color).add_to(map)

        prev_location = location
    folium_static(map)

# Run the app
if __name__ == '__main__':
    app()
