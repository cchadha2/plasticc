import pandas as pd
import numpy as np
import time

from contextlib import contextmanager
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

# Initialise all inputs
submission_file_name = 'preds/lgb_test_preds_1.8.csv'

# Data
test_dataset = 'processed_test_1.8.csv'
training_dataset = 'processed_train_1.8.csv'
train_df = pd.read_csv('output/' + training_dataset, index_col=False)

le = LabelEncoder()
train_df.target = le.fit_transform(train_df.target)
print('Number of training classes: {}'.format(len(le.classes_.tolist())))
        
# Dataframes for predictions
sub_df = pd.read_csv(r"output/"+ test_dataset,usecols=[0])
for x in le.classes_.tolist():
    sub_df['class_' + str(x)] = 0
sub_df['class_99'] = 0

# Training parameters
num_folds = 10
stratified = False
SEED = 1001
early_rounds = 50
importance_save = True

# Model parameters
params = {
        'device': 'cpu', 
        'objective': 'multiclass', 
        'num_class': 14, 
        'boosting_type': 'gbdt', 
        'n_jobs': -1, 
        'max_depth': 7, 
        'n_estimators': 500, 
        'subsample_freq': 2, 
        'subsample_for_bin': 5000, 
        'min_data_per_group': 100, 
        'max_cat_to_onehot': 4, 
        'cat_l2': 1.0, 
        'cat_smooth': 59.5, 
        'max_cat_threshold': 32, 
        'metric_freq': 10, 
        'verbosity': -1, 
        'metric': 'multi_logloss', 
        'xgboost_dart_mode': False, 
        'uniform_drop': False, 
        'colsample_bytree': 0.5, 
        'drop_rate': 0.173, 
        'learning_rate': 0.0267, 
        'max_drop': 5, 
        'min_child_samples': 10, 
        'min_child_weight': 100.0, 
        'min_split_gain': 0.1, 
        'num_leaves': 7, 
        'reg_alpha': 0.1, 
        'reg_lambda': 0.00023, 
        'skip_drop': 0.44, 
        'subsample': 0.75,
        'seed': SEED}

def multi_weighted_logloss(y_true, y_preds):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    # class_weights taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    if y_true.shape[1] > 14:
        classes.append(99)
        class_weight[99] = 2
        y_true = y_true[:, :-1]

    if y_preds.shape[1] > 14:
        y_preds = y_preds[:, :-1]

    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_preds = np.clip(a=y_preds, a_min=1e-15, a_max=1 - 1e-15)

    # Transform to log
    y_p_log = np.log(y_preds)

    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_true * y_p_log, axis=0)        
    
    # Get the number of positives for each class
    nb_pos = y_true.sum(axis=0).astype(float)

    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss

def lgb_multi_weighted_logloss(y_true, y_preds):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    # class_weights taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')

    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return 'wloss', loss, False

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f} mins".format(title, (time.time() - t0)/60))

def smoteAdataset(Xig_train, yig_train, Xig_test, yig_test):
    
        
    sm=SMOTE(random_state=2)
    Xig_train_res, yig_train_res = sm.fit_sample(Xig_train, yig_train.ravel())

        
    return Xig_train_res, pd.Series(yig_train_res), Xig_test, pd.Series(yig_test)

with timer("Run LightGBM with kfold"):
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=SEED)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=SEED)

    # Create arrays and dataframes to store results
    sub_preds = np.zeros((sub_df.shape[0], len(le.classes_)))
    feats = [f for f in train_df.columns if f not in ['target','object_id']]

    w = train_df['target'].value_counts()
    weights = {i : np.sum(w) / w[i] for i in w.index}

    for cindex in train_df.columns:
        train_df.loc[:,cindex]=np.float64(train_df.loc[:,cindex])

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['target'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
        val_x, val_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

        trn_xa, train_y, val_xa, val_y=smoteAdataset(train_x.values, train_y.values, val_x.values, val_y.values)
        train_x=pd.DataFrame(data=trn_xa, columns=train_x.columns)
        val_x=pd.DataFrame(data=val_xa, columns=val_x.columns)

        print("{}/{} folds".format(n_fold+1, num_folds))
        print("Starting training. Train shape: {}".format(train_df.shape))

        clf = lgb.LGBMClassifier(**params)

        clf.fit(
            train_x,
            train_y,
            eval_set=[(train_x, train_y), (val_x, val_y)],
            eval_metric=lgb_multi_weighted_logloss,
            early_stopping_rounds= early_rounds,
            verbose=100,
            sample_weight=train_y.map(weights)
        )
        
        print("Predicting")
        print("__________")
        test_dfs = pd.read_csv('output/' + test_dataset, chunksize=1000000, index_col=False)
        sub_preds_list = [clf.predict_proba(test_df[feats])/folds.n_splits for test_df in test_dfs]

        sub_preds += np.vstack(sub_preds_list)

    # preds_99 = np.ones((sub_preds.shape[0],1))
    # for i in range(sub_preds.shape[1]):
    #     preds_99[:,0] *= 1 - sub_preds[:, i]
    # sub_preds = np.hstack((sub_preds, preds_99))

    # Dampen prediction of class_99 with constant
    preds_99 = np.ones((sub_preds.shape[0],1))
    preds_99 = 0.14*(1-np.max(sub_preds,axis=1).reshape(-1,1))
    sub_preds = np.hstack((sub_preds, preds_99))
    del clf

    sub_df.iloc[:, 1:] = sub_preds
    sub_df.to_csv(submission_file_name, index= False)
    print('Completed predicting and saved submission CSV {}'.format(sub_df.shape))