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
#%%
OMP_NUM_THREADS=1
#%%
#get all the data loaded in and take a look
newslet = pd.read_csv("newsletter_stats.csv", header=0)
spkr_eval = pd.read_csv("speaker_eval.csv",header=0)
spkr_view_stats =pd.read_csv("speaker_viewer_stats.csv", header=0)

# %% data merging and cleaning 
#Need to merge all these into one file using the Speaker column 
data_merge = newslet.merge(spkr_eval.merge(spkr_view_stats, on='Speaker'), 
on='Speaker')
#data_merge.info()
#drop the date column
data_merge = data_merge.drop(['Date'], axis = 1)
# need to remove all % signs and convert those columns to decimals essentially divide by 100. 
data_merge = data_merge.replace(to_replace ='%', value = '', regex = True)
#replace missing value in Average Percentage Viewed 
data_merge.at[23,'Average Percentage Viewed'] = 15
#data_merge['Average Percentage Viewed'].mean()
convert_dict={'Average Percentage Viewed': float}
data_merge = data_merge.astype(convert_dict)

#%% data scaling 
#need to scale everything aside from the average, run the fit first 
data_merge.iloc[:,2:36] = scale.fit_transform(data_merge.iloc[:,2:36])
#copy it to clipboard in a excel format so I can put it into a table. 

#Try minmax between 0 and 1 instead, this worked better
#mms = MinMaxScaler()
#data_merge.iloc[:,2:36] = mms.fit_transform(data_merge.iloc[:,2:36])
#data_merge.to_clipboard(excel=True)

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
ave_rate_spk = speaker_perc.mean(axis=0)

speaker_perc['Strongly Agree_5_yt'] = speaker_perc['Strongly Agree_5_yt'].astype(int)

mean_spkr_rate = speaker_perc.mean(axis=1)

speaker_perc = pd.concat([speaker_perc,mean_spkr_rate], axis=1,join='inner')

#copy to clipboard thats ready to paste into excel 
speaker_perc.to_clipboard(excel=True)

speaker_perc.rename(columns = {0:"ave"}, inplace=True)

summary_speaker_prec = speaker_perc.describe()
#Speak to the 3rd/75% upper quartile just 5 speakers, in terms of average feedback in the 
#strongly agree category.
summary_speaker_prec

#%% Not sure I needed to do any of this...but converting data types from object to float
# Let's do some clustering but need to convert everything to a int, likely need to write a 
# function to make that easier. 

#start here...need to convert these all to int, to do the clustering

#selecting on column name 
agree = data_merge.loc[:, data_merge.columns.str.contains('Agree')]
#creating a index based on column type
sel = agree.select_dtypes(include='object').columns
#passing the index back into the original dataset to change the variable types
data_merge[sel] = data_merge[sel].astype("float")
#average percentage viewed 
#data_merge['Average Percentage Viewed'] = data_merge.Average Percentage Viewed.astype(float)

#%%

data_merge.dtypes

breakpoint

#%% Clustering data
cluster_data = data_merge.iloc[:,2:36]
#replaced missing data point and change to a array
cluster_data = cluster_data.to_numpy()
cluster_data[10,29] = .05

#%% clustering algo 
init_kmeans = KMeans(
    init="random",
    n_clusters=7,
    n_init=10,
    max_iter=300,
    random_state=1518
    )
#changed the data to a numby array

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
#%%
plt.style.use("fivethirtyeight")
plt.plot(range(1, 12), sse)
plt.xticks(range(1, 12))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()
#%% Capturing the predicted lables. 
label = init_kmeans.fit_predict(cluster_data)

#%% #x cluster no.,y used to generate cluster, z is label index
def label_index_1(x,y,z):
    #x cluster no.,y used to generate cluster, z is label index
    filtered_label = y[z == x]
    return filtered_label 

#%%
label_n = []

for k in range(0,7):
    output = label_index_1(k,cluster_data,label)
    plt.scatter(output[:,0] , output[:,1], label = k)
    plt.legend()
   


# %%
xxx = label_index_1(1,cluster_data,label)

#plt.savefig('speaker_cluster.png')

# %%
label_pd = pd.DataFrame(label)
label_pd.to_clipboard(excel=True)
# %%
filtered_label0[:,0]