#%%
from datetime import date
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from pandas.core.reshape.merge import merge_ordered
#%%
#get all the data loaded in and take a look
newslet = pd.read_csv("newsletter_stats.csv", header=0)
spkr_eval = pd.read_csv("speaker_eval.csv",header=0)
spkr_view_stats =pd.read_csv("speaker_viewer_stats.csv", header=0)
# %%
#Need to merge all these into one file using the Speaker column 
data_merge = newslet.merge(spkr_eval.merge(spkr_view_stats, on='Speaker'), on='Speaker')
#data_merge.info()
# need to remove all % signs and convert those columns to decimals essentially divide by 100. 

#ok this worked
abc=data_merge['Average Percentage Viewed'].map(lambda x: str(x)[:-1])


#but this is way easier!
#data_merge = data_merge.replace(to_replace ='%', value = '', regex = True)
data_merge.sort
# %%
#Summary Stats 
summary_stat = data_merge.describe()
#Distribution of Speakers 
xx =  data_merge.loc[:, data_merge.columns.str.startswith('Strongly Agree')]
xy = data_merge.loc[:,'Speaker']

speaker_perc = pd.concat([xy,xx],axis=1, join='inner')


#need to find the average of rows 

#%%
ave_rate_spk = speaker_perc.mean(axis=0)

speaker_perc['Strongly Agree_5_yt'] = speaker_perc['Strongly Agree_5_yt'].astype(int)
 
speaker_perc.dtypes

mean_spkr_rate = speaker_perc.mean(axis=1)

speaker_perc = pd.concat([speaker_perc,mean_spkr_rate], axis=1,join='inner')

speaker_perc.rename(columns = {0:"ave"}, inplace=True)

summary_speaker_prec = speaker_perc.describe()
#Speak to the 3rd/75% upper quartile just 5 speakers, in terms of average feedback in the 
#strongly agree category. 

#%%
# Let's do some clustering but need to convert everything to a int, likely need to write a 
# function to make that easier. 

#start here...need to convert these all to int, to do the clustering

convert_dict={'n_eval_rsp': int} 
data_merge = data_merge.astype(convert_dict)

data_merge.dtypes

#%%
Speaker                                        object
Date Sent                                      object
#_nw_Opens                                      int64
#_nl_Clicks                                     int64
n_eval_rsp                                    float64
Strongly Agree_1_vi                            object
Somewhat agree_1_vi                            object
Neither agree nor disagree_1_vi                object
Somewhat disagree_1_vi                         object
Strongly disagree_1_vi                         object
Strongly Agree_2_rl                            object
Somewhat agree_2_rl                            object
Neither agree nor disagree_2_rl                object
Somewhat disagree_2_rl                         object
Strongly disagree_2_rl                         object
Strongly Agree_3_qa                            object
Somewhat agree_3_qa                            object
Neither agree nor disagree_3_qa                object
Somewhat disagree_3_qa                         object
Strongly disagree_3_qa                         object
Strongly Agree_4_qa                            object
Somewhat agree_4_fp                            object
Neither agree nor disagree_4_fp                object
Somewhat disagree_4_fb                         object
Strongly disagree_4_fb                         object
Strongly Agree_5_yt                            object
Somewhat agree_5_yt                            object
Neither agree nor disagree_5_yt                object
Somewhat disagree_5_yt                         object
Strongly disagree_5_yt                         object
Date                                           object
# of Innovation Lab (zoom) attendees            int64
# Youtube live streamers                      float64
# of Total Youtube views (as of 6/11/2021)      int64
# of unique viewers in the first 90 days        int64
Total Watch Time (hrs)                        float64
Average Percentage Viewed                      object
dtype: object