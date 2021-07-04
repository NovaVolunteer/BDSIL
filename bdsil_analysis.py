#%%
import pandas as pd
import numpy as np
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
data_merge = data_merge.replace(to_replace ='%', value = '', regex = True)

# %%
range(1,10)
# %%
#apply pushing something across all the columns the data.frame 
# and lambda is a shortcut for doing a function. 
def rstr(df): return df.shape, df.apply(lambda x: [x.unique()])


# %%
