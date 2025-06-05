#%%
#import docx
from datetime import date
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from pandas.core.reshape.merge import merge_ordered
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
scale = StandardScaler()
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pdb
import openpyxl
#import plotnine

#%% Untested function
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import re

def prepare_speaker_data(filepath: str) -> pd.DataFrame:
    """
    Loads and preprocesses speaker data from a CSV file.
    Cleans column names, scales numeric data, computes impact scores,
    and returns a DataFrame ready for clustering.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned and processed data suitable for clustering.
    """
    # Load data
    data_merge = pd.read_csv(filepath)

    # Clean column names: lowercase and remove special characters
    cleaned_columns = [
        re.sub(r'[^a-z0-9_]', '_', col.lower()) for col in data_merge.columns
    ]
    data_merge.columns = cleaned_columns

    # Remove commas and percent signs from the data
    data_merge = data_merge.replace({',': '', '%': ''}, regex=True)

    # Convert appropriate columns to float
    total_cols = data_merge.shape[1]
    data_merge.iloc[:, 2:total_cols - 1] = data_merge.iloc[:, 2:total_cols - 1].astype(float)

    # Scale numeric values using MinMaxScaler
    mms = MinMaxScaler()
    data_merge.iloc[:, 2:total_cols] = mms.fit_transform(data_merge.iloc[:, 2:total_cols])

    # Identify impact-related columns
    impact_indices = [0] + list(range(2, 6)) + [total_cols - 2, total_cols - 1]
    impact_df = data_merge.iloc[:, impact_indices].copy()
    impact_df['total'] = impact_df.iloc[:, 1:].sum(axis=1)

    # Drop known bad row, if it exists
    if 16 in impact_df.index:
        impact_df = impact_df.drop(index=16)

    # Average of "strongly agree" columns
    agree_cols = [col for col in data_merge.columns if col.startswith('strongly_agree')]
    speaker_col = 'speaker' if 'speaker' in data_merge.columns else data_merge.columns[0]
    speaker_perc = pd.concat([data_merge[speaker_col], data_merge[agree_cols]], axis=1)
    speaker_perc['ave_strongly_agree'] = speaker_perc.iloc[:, 1:6].mean(axis=1)

    # Add to impact_df
    impact_df = pd.concat([impact_df, speaker_perc['ave_strongly_agree']], axis=1)

    # Return only numeric columns for clustering
    return impact_df.drop(columns=['speaker', 'total'], errors='ignore')




#%% Load the data
#get all the data loaded in and take a look
data_merge = pd.read_csv("2025bdsil.csv")


#%% data scaling 

# remove coma and % from the data_merge
data_merge = data_merge.replace(to_replace =',', value = '', regex = True)
data_merge = data_merge.replace(to_replace ='%', value = '', regex = True)
#convert to float aside from first column
data_merge.iloc[:,2:33] = data_merge.iloc[:,2:33].astype(float)

#%%
#Try minmax between 0 and 1 instead, this worked better
mms = MinMaxScaler()
data_merge.iloc[:,2:34] = mms.fit_transform(data_merge.iloc[:,2:34])

#data_merge.to_excel('speaker_stats.xlsx',sheet_name = 'sheet1', index=False)



#%%
###### CREATE IMPACT SCORE - ALL SOCIAL MEDIA + NEWSLETTER COLUMNS ######
# Create a list of column indices you want to include
col_indices = [0] + list(range(2, 6)) + [31, 32]

# Use .iloc to select columns by index
impact_df = data_merge.iloc[:, col_indices].copy()

# Sum the numeric columns (excluding the first column, which is likely non-numeric)
impact_df['total'] = impact_df.iloc[:, 1:].sum(axis=1)

#%%
#drop row 16
impact_df = impact_df.drop(index=16)

#%%
#gathering various columns to  
xx =  data_merge.loc[:, data_merge.columns.str.startswith('Strongly Agree')]
xy = data_merge.loc[:,'Speaker']

speaker_perc = pd.concat([xy,xx],axis=1, join='inner')


#need to find the average of rows 

#%% generating strongly agree average 


#sum columns zero five then divide by 5
speaker_perc["ave_strongly_agree"] = speaker_perc.iloc[:,1:6].mean(axis=1)


#%%
#add the average to the impact_df
impact_df = pd.concat([impact_df, speaker_perc["ave_strongly_agree"]], axis=1)

#%% Clustering data
cluster_data = impact_df.drop(columns=['Speaker',"total"])

#replaced missing data point and change to a array
#cluster_data = cluster_data.to_numpy()
#cluster_data[10,29] = .05

#%%
#remove row 16
cluster_data = np.delete(cluster_data, 16, axis=0)

#%% clustering algo 
init_kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init=10,
    max_iter=300,
    random_state=1518
    )
#changed the data to a numby array
#%%
init_kmeans.fit(cluster_data)
init_kmeans.inertia_
init_kmeans.n_iter_

#%% Need to optimize the data, looks like 6, we don't pass in the cluster
#argument here because we will for loop that below. 
kmeans_args = {
"init": "random",
"n_init": 7,
"max_iter": 300,
"random_state": 1518,
}

#frame for the standard error output 
sse= []
#simple 'for loop' to run through the options, function would be better but short on time.
for k in range(1, 12):
    kmeans = KMeans(n_clusters=k, **kmeans_args) #**special chara that allows to 
    #pass multiple arguments
    kmeans.fit(cluster_data)
    sse.append(kmeans.inertia_)


#%% Capturing the predicted lables. 
label = init_kmeans.fit_predict(cluster_data)

#%%
#remove last row from impact_df
impact_df = impact_df.drop(index=16)
#merge labels to the impact_df
impact_df['clusters'] = label

#%%
#rename %Strongly Agree to ave_strong_agree
impact_df.rename(columns={'Average Percentage Viewed':'average_perc_viewed'}, inplace=True)
impact_df.rename(columns={'# Youtube live streamers':'youtube_streamers'}, inplace=True)

#%%
import matplotlib.pyplot as plt

#%%
impact_df.info()

#%%
#remove all special characters from the dataframe columns
impact_df.columns = impact_df.columns.str.replace('[^a-zA-Z0-9]', '_', regex=True)
#remove spaces from the column names
impact_df.columns = impact_df.columns.str.replace(' ', '_', regex=True)
#make all columns lower case
impact_df.columns = impact_df.columns.str.lower()


#%%
# why is this not working?

cmap = ListedColormap(["orange","red","blue"], name='clusters', N=None)
# Create a scatter plot for the current cluster 
import matplotlib.pyplot as plt

# Example: assuming label is a list or Series of cluster labels and cmap is defined
xx = plt.scatter(
    x=impact_df["ave_strongly_agree"],
    y=impact_df["__of_clicks"],
    c=impact_df["clusters"],# color based on cluster labels
    cmap=cmap,              # specify the colormap
    edgecolor='black',
    marker='o',
    s=150,
    alpha=0.5
)

plt.title("Clusters of Speaker Reviews")
plt.xlabel("Ave_Strong_Agree")
plt.xticks(rotation=45)
plt.ylabel("YouTube Streamers")

# Create legend from scatter elements
legend1 = plt.legend(*xx.legend_elements(), loc="lower left", title="Clusters")
plt.gca().add_artist(legend1)

plt.show()

#%%
print(impact_df["clusters"].unique())
impact_df["clusters"] = impact_df["clusters"].astype(int)
# %%python3 -m venv /path/to/new/virtual/environment
fig
# %%
import plotly.express as px
fig = px.scatter(impact_df, x="ave_strongly_agree", y="__youtube_live_streamers", color="clusters")
# %%

impact_df.info()
# %%
