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
#import plotnine
#%%
OMP_NUM_THREADS=1
#%%
#get all the data loaded in and take a look
data_merge = pd.read_csv("spkr_stats_24.csv")

# %% data merging and cleaning 
#Need to merge all these into one file using the Speaker column 
#data_merge = newslet.merge(spkr_eval.merge(spkr_view_stats, on='Speaker'), 
#on='Speaker')
#data_merge.info()

# need to remove all % signs and convert those columns to decimals essentially divide by 100. 
data_merge = data_merge.replace(to_replace ='%', value = '', regex = True)
#replace missing value in Average Percentage Viewed 
#data_merge.at[23,'Average Percentage Viewed'] = 15
data_merge['Average Percentage Viewed'].mean()
convert_dict={'Average Percentage Viewed': float}
data_merge = data_merge.astype(convert_dict)

#%% data scaling 

# remove coma and % from the data_merge
data_merge = data_merge.replace(to_replace =',', value = '', regex = True)
data_merge = data_merge.replace(to_replace ='%', value = '', regex = True)
#convert to float aside from first column
data_merge.iloc[:,1:34] = data_merge.iloc[:,1:34].astype(float)

#%%
#Try minmax between 0 and 1 instead, this worked better
mms = MinMaxScaler()
data_merge.iloc[:,1:34] = mms.fit_transform(data_merge.iloc[:,1:34])

#data_merge.to_excel('speaker_stats.xlsx',sheet_name = 'sheet1', index=False)

# %% generating summary stats and gathering strongly agree columns
#Summary Stats 
summary_stat = data_merge.describe()
#transpose to make the table more readable
summary_stat = summary_stat.transpose()
#copy it to excel clipboard
#%%
data_merge_1.to_excel('spkr_stats_24.xlsx',sheet_name = 'sheet2', index=False)
#%%

#gathering various columns to  
xx =  data_merge.loc[:, data_merge.columns.str.startswith('Strongly Agree')]
xy = data_merge.loc[:,'Speaker']

speaker_perc = pd.concat([xy,xx],axis=1, join='inner')

#need to find the average of rows 

#%% generating strongly agree average 

#generate the means 
ave_rate_spk = speaker_perc.mean(axis=1)
#%%


#speaker_perc['Strongly Agree_5_yt'] = speaker_perc['Strongly Agree_5_yt'].astype(int)

mean_spkr_rate = speaker_perc.mean(axis=1)

speaker_perc = pd.concat([speaker_perc,mean_spkr_rate], axis=1,join='inner')

#copy to clipboard thats ready to paste into excel 
speaker_perc.rename(columns = {0:"ave"}, inplace=True)

summary_speaker_prec = speaker_perc.describe()
#Speak to the 3rd/75% upper quartile just 5 speakers, in terms of average feedback in the 
#strongly agree category.
summary_speaker_prec

speaker_perc.to_excel('speaker_perc_1.xlsx',sheet_name = 'sheet1', index=False)
#%%

data_merge.dtypes

#%% Clustering data
cluster_data = data_merge.iloc[0:19,1:34]

#replaced missing data point and change to a array
cluster_data = cluster_data.to_numpy()
#cluster_data[10,29] = .05

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
"n_init": 6,
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

# %% checking on missing data, which I had and inifity which I didn't 
np.any(np.isnan(cluster_data))#one missing data point so didn't work
#np.all(np.isfinite(cluster_data))

#%% Capturing the predicted lables. 
label = init_kmeans.fit_predict(cluster_data)


#%%
data_merge_1=data_merge.loc[0:19,]

data_merge_1['clusters']=label
#%%
#rename %Strongly Agree to ave_strong_agree
data_merge_1.rename(columns={'% Strongly Agree':'ave_strong_agree'}, inplace=True)
data_merge_1.rename(columns={'Average Percentage Viewed':'average_perc_viewed'}, inplace=True)
data_merge_1.rename(columns={'#_Youtube_live_streamers':'youtube_streamers'}, inplace=True)

#%%
import matplotlib.pyplot as plt

#%%
#remove the space in the column name
data_merge_1.columns = data_merge_1.columns.str.replace(' ', '_')

#%%
cmap = ListedColormap(["orange","red","blue"], name='Clusters', N=None)
xx=plt.scatter(x=data_merge_1.ave_strong_agree, y=data_merge_1.youtube_streamers,
            s=150, c=label, cmap=cmap, alpha=0.5)
plt.title("Clusters of Speaker Reviews")
plt.xlabel("YouTube Streamers")
plt.ylabel("Average Percentage Viewed")
legend1=plt.legend(*xx.legend_elements(),loc="lower left", title="Clusters")
plt.show()

# %%python3 -m venv /path/to/new/virtual/environment
fig
# %%
import plotly.express as px
fig = px.scatter(data_merge_1, x="ave_strong_agree", y="average_perc_viewed", color="clusters")
# %%
