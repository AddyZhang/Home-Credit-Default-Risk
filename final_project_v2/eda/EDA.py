import pandas as pd
import matplotlib.pyplot as plt


class EDA:
    def __init__(self):
        pass

    @staticmethod
    def info(df):
       """todo"""

    @staticmethod
    def missing_table(df):
        """Display the number of missing values and percentage in each features"""
        missing_series = df.isnull().sum()
        missing_percentage = df.isnull().sum() / len(df) * 100
        missing_df = pd.concat([missing_series, missing_percentage], axis=1)
        missing_df.columns = ['missing_values', 'missing_percentage(%)']
        missing_df.sort_values(by='missing_percentage(%)', ascending=False, inplace=True)
        return missing_df

    @staticmethod
    def plot_target(target):
        """
        target: target variable (dataframe type).
        Use histogram to observe if the target is balanced or not
        :return: none
        """
        target.plot.hist(title='Histogram')
        plt.xlabel('Target')
        plt.ylabel('Frequency')
