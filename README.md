# Home-Credit-Default-Risk
This is a kaggle project. You can find detailed problem descripition and datasets from this link: https://www.kaggle.com/c/home-credit-default-risk
This project contains a large amount of missing values for certain features and target variables are not balanced.
We basically select features with missing values less than 15% as feature selection datasets. Then we select features with more than one vote from those feature selection methods. 
We thought there are still important features existed with missing values more than 85%. Then we use find features with high correlation to target. 
We combine those feautures and also construct new features based on our domain knowledge.
We use XGBoost and LightGBM models to train since the two models could handle missing values internally; When the model splits at a node, it allocates the missing values or zero values to whichever sides reduce the loss most.
Hyparameter Tuning using grid search
Train new models with the best hyperparameters found from the last step.
