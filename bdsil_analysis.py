#%%
import docx
from datetime import date
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
import plotnine
#%%
OMP_NUM_THREADS=1
#%%
#get all the data loaded in and take a look
data_merge = pd.read_csv("spkr_stats.csv")

# %% data merging and cleaning 
#Need to merge all these into one file using the Speaker column 
#data_merge = newslet.merge(spkr_eval.merge(spkr_view_stats, on='Speaker'), 
#on='Speaker')
#data_merge.info()

# need to remove all % signs and convert those columns to decimals essentially divide by 100. 
#data_merge = data_merge.replace(to_replace ='%', value = '', regex = True)
#replace missing value in Average Percentage Viewed 
#data_merge.at[23,'Average Percentage Viewed'] = 15
#data_merge['Average Percentage Viewed'].mean()
convert_dict={'Average Percentage Viewed': float}
data_merge = data_merge.astype(convert_dict)

#%% data scaling 
#need to scale everything aside from the average, run the fit first 
#data_merge.iloc[:,2:36] = scale.fit_transform(data_merge.iloc[:,2:36])
#copy it to clipboard in a excel format so I can put it into a table. 

#Try minmax between 0 and 1 instead, this worked better
mms = MinMaxScaler()
data_merge.iloc[:,2:36] = mms.fit_transform(data_merge.iloc[:,2:36])
data_merge.to_clipboard(excel=True)

# %% generating summary stats and gathering strongly agree columns
#Summary Stats 
summary_stat = data_merge.describe()
#transpose to make the table more readable
summary_stat = summary_stat.transpose()
#copy it to excel clipboard
summary_stat.to_clipboard(excel=True)


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
speaker_perc.to_clipboard(excel=True)

speaker_perc.rename(columns = {0:"ave"}, inplace=True)

summary_speaker_prec = speaker_perc.describe()
#Speak to the 3rd/75% upper quartile just 5 speakers, in terms of average feedback in the 
#strongly agree category.
summary_speaker_prec


#%%

data_merge.dtypes

#%% Clustering data
cluster_data = data_merge.iloc[0:20,1:33]
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
#simple for loop to run through the options, function would be better but short on time.
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
data_merge_1['ave_strong_agree']=ave_rate_spk.loc[0:19,]
#%%
#plt.scatter(data_merge_1.ave_strong_agree, data_merge_1.Average Percentage Viewed, c=labels, cmap='viridis')
#plt.show()
from matplotlib import cm
from matplotlib.colors import ListedColormap

#%%
cmap = ListedColormap(["red","orange","blue"], name='Clusters', N=None)
data_merge_1.plot(kind='scatter', x='ave_strong_agree', y='# of Innovation Lab (zoom) attendees', 
                  s=150, c=label, cmap=cmap, title="Clusters of Engagement")
plt.show()

# %%python3 -m venv /path/to/new/virtual/environment
fig
# %%
