from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
import numpy as np
import pandas as pd

from eda import EDA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
# import logging
#
# LOG_FORMAT = "%(Levelname)s %(asctime)s - %(message)s"
# logging.basicConfig(filename="/Users/Zhang/Documents/Predictive Analytics/Final_Project/FP.Log",
#                     level=logging.DEBUG,
#                     format=LOG_FORMAT,
#                     filemode='w')
# logger = logging.getLogger()


class FeatureSelector:

    def __init__(self, df):
        self.df = df

        self.feature_train = None
        self.target_train = None

        self.f_support = None
        self.rfe_support = None
        self.emb_lr = None
        self.emb_rf_sup = None
        self.emb_xgb_sup = None
        self.vote_df = None
        self.features_vote = None

    def feature_thresh(self, n=15):
        """
        Before feature engineering, it is crucial to select features with less missing values because some machine
        learning algorithms used by feature selection cannot handle a lot of missing values.
        :param missing_df: dataframe created by eda.missing_table
        :param n: threshold
        :return: feature_train, target_train
        """
        missing_df = EDA().missing_table(self.df)
        feature_idx = missing_df[missing_df['missing_percentage(%)'] <= n].index
        data = self.df[feature_idx]

        data1 = data.dropna()

        self.target_train = data1['TARGET']

        data_train = data1.drop('TARGET', axis=1)
        self.feature_train = pd.get_dummies(data_train, drop_first=True)

        return self.feature_train, self.target_train

    # def show_features(self):
    #     feature = self.feature_train.loc[:self.f_support].columns.tolist()

    def get_feature(self, model):
        model.fit(self.feature_train, self.target_train)
        return model.get_support()

    def anova_f(self, k=10):
        """
        x = feature_train by default
        y = target_train by default
        k = number of best features to be selected
        """
        print("Its anova")
        f_class = SelectKBest(f_classif, k)
        return self.get_feature(f_class)

    def rfe(self, k=10):
        print("its rfe")
        estimator = LogisticRegression(solver='lbfgs', max_iter=4000)
        rfe_selector = RFE(estimator,
                           n_features_to_select=k,
                           step=10,
                           verbose=5)
        return self.get_feature(rfe_selector)

    def lasso(self, k=10):
        print("its lasso")
        est = LogisticRegression(solver='liblinear', penalty='l1')
        embbed_lr = SelectFromModel(est, max_features=k)
        return self.get_feature(embbed_lr)

    def rf(self, k=10):
        print("its random forest")
        mdl = RandomForestClassifier(n_estimators=100)
        emb_rf_sel = SelectFromModel(mdl, max_features=k)
        return self.get_feature(emb_rf_sel)

    def xgb(self, k=10):
        print("its xgb")
        xgbc = XGBClassifier(n_estimators=500, learning_rate=0.05,
                             num_leaves=32, colsample_bytree=0.2,
                             reg_alpha=3, reg_lambda=1,
                             min_split_gain=0.01, min_child_weight=40)
        emb_xgb_sel = SelectFromModel(xgbc, max_features=k)
        return self.get_feature(emb_xgb_sel)

    def vote(self):
        print("it vote")
        if self.target_train is None:
            self.target_train = self.feature_thresh()
        # self.f_support = self.anova_f()
        if self.f_support is None:
            self.anova_f()
        if self.emb_lr is None:
            self.lasso()
        # self.emb_lr = self.lasso()
        self.rfe_support = self.rfe()
        self.emb_rf_sup = self.rf()
        self.emb_xgb_sup = self.xgb()
        self.vote_df = pd.DataFrame({"Feature": self.feature_train.columns,
                                "Random Forest": self.emb_rf_sup,
                                "Lasso": self.emb_lr,
                                "RFE": self.rfe_support,
                                "XGBoost": self.emb_xgb_sup,
                                "f_classif": self.f_support})
        self.vote_df['total'] = np.sum(self.vote_df, axis=1)
        self.vote_df = self.vote_df.sort_values(['total', 'Feature'], ascending=False)
        return self.vote_df

    def get_fea_vote(self, n=1):
        """

        :param n: the criteria by number of vote
        :return: features selected by the number of vote (dataframe)
        """
        if self.vote_df is None:
            self.vote_df = self.vote()
        feature_model_idx = self.vote_df[self.vote_df.total > n]['Feature']
        self.features_vote = self.feature_train[feature_model_idx]

        return self.features_vote
