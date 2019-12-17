import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from feature_select import FeatureSelector


class Sampling(FeatureSelector):
    def __init__(self, df):
        super().__init__(df)

        self.predictor = None
        self.target = None

        self.x_train = None
        self.y_train = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

    def get_tar_pre(self):
        feature_train, self.target = super().feature_thresh()
        self.predictor = super().features_vote

    def split(self):
        self.x_train, self.X_test, self.y_train, self.Y_test = train_test_split(self.predictor,
                                                                                self.target,
                                                                                test_size=0.3,
                                                                                random_state=1)
        return self.x_train, self.X_test, self.y_train, self.Y_test

    def subsampling(self):
        self.x_train, _, self.y_train,_ = self.split()
        X = pd.concat([self.X_train, self.y_train], axis=1)
        default_1 = X[X.iloc[:, -1] == 1]
        non_default_0 = X[X.iloc[:, -1] == 0]
        non_default_under = resample(non_default_0,
                                     replace=False,
                                     n_samples=len(default_1),
                                     random_state=1)
        downsampled = pd.concat([non_default_under, default_1])
        self.X_train = downsampled.iloc[:,-1]
        self.Y_train = downsampled.iloc[:,:-1]
        return self.X_train, self.Y_train

    def oversampling(self):
        """to do"""


