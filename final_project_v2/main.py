import pandas as pd
import numpy as np
import logging

import warnings
warnings.filterwarnings('ignore')
from feature_select import FeatureSelector
from eda import EDA

df = pd.read_csv('application_train.csv',index_col = 0)

missing_df = EDA.missing_table(df)
EDA.plot_target(df['TARGET'])

df_obj = FeatureSelector(df)
feature_train, target_train = df_obj.feature_thresh(n = 11)
print(feature_train.head())

print(df_obj.anova_f())

vote_df = df_obj.vote()
# vote_df=df_obj.vote()

# print(vote_df.head())