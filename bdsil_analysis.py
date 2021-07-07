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

#%%
#get all the data loaded in and take a look
newslet = pd.read_csv("newsletter_stats.csv", header=0)
spkr_eval = pd.read_csv("speaker_eval.csv",header=0)
spkr_view_stats =pd.read_csv("speaker_viewer_stats.csv", header=0)

# %%
#Need to merge all these into one file using the Speaker column 
data_merge = newslet.merge(spkr_eval.merge(spkr_view_stats, on='Speaker'), on='Speaker')
#data_merge.info()
#drop the date column
data_merge = data_merge.drop(['Date'], axis = 1)
# need to remove all % signs and convert those columns to decimals essentially divide by 100. 
#but this is way easier!
data_merge = data_merge.replace(to_replace ='%', value = '', regex = True)

#replace missing value in Average Percentage Viewed 

data_merge.at[23,'Average Percentage Viewed'] = 15

#data_merge['Average Percentage Viewed'].mean()
convert_dict={'Average Percentage Viewed': float}
data_merge = data_merge.astype(convert_dict)

#%%
#need to scale everything aside from the average, run the fit first 
#data_merge.iloc[:,3:35] = scale.fit_transform(data_merge.iloc[:,3:35])
#data_merge.iloc[:,3:35] = scale.transform(data_merge.iloc[:,3:35]) 
#copy it to clipboard in a excel format so I can put it into a table. 


#Try minmax between 0 and 1 instead, this worked better
mms = MinMaxScaler()
data_merge.iloc[:,3:35] = mms.fit_transform(data_merge.iloc[:,3:35])
data_merge.iloc[:,3:35] = scale.transform(data_merge.iloc[:,3:35])
data_merge.to_clipboard(excel=True)




# %%
#Summary Stats 
summary_stat = data_merge.describe()

summary_stat = summary_stat.transpose()

#Distribution of Speakers 
xx =  data_merge.loc[:, data_merge.columns.str.startswith('Somewhat Agree')]
xy = data_merge.loc[:,'Speaker']

speaker_perc = pd.concat([xy,xx],axis=1, join='inner')


#need to find the average of rows 

#%%
ave_rate_spk = speaker_perc.mean(axis=0)

speaker_perc['Strongly Agree_5_yt'] = speaker_perc['Strongly Agree_5_yt'].astype(int)
 
speaker_perc.dtypes

mean_spkr_rate = speaker_perc.mean(axis=1)

speaker_perc = pd.concat([speaker_perc,mean_spkr_rate], axis=1,join='inner')

#copy to clipboard thats ready to paste into excel 
speaker_perc.to_clipboard(excel=True)

speaker_perc.rename(columns = {0:"ave"}, inplace=True)

summary_speaker_prec = speaker_perc.describe()
#Speak to the 3rd/75% upper quartile just 5 speakers, in terms of average feedback in the 
#strongly agree category.

#%%
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


#data_merge.dtypes

#%%
