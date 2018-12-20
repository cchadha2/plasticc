import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import itertools

from contextlib import contextmanager
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import pymongo

from data_process.feature_engineering_lgb_tsfresh import data_process

# Data
data = 'data/'
output = 'output/'
train_name = 'processed_train_tsfresh.csv'

start = time.time()
train = pd.read_csv(data + 'training_set.csv', index_col=False)
print('Processing train dataset. train shape: {}'.format(train.shape))
train_df = data_process(train)
print('Processed train. Train shape: {};'.format(train_df.shape))
train_df.to_csv(output + train_name, index=False)
end = time.time()
print('Time taken to process train set: {:.2f}s'.format(end-start))
print(train_df.head())

le = LabelEncoder()
train_df.target = le.fit_transform(train_df.target)
print('Number of training classes: {}'.format(len(le.classes_.tolist())))

# Feature importance dataframes
fold_importance_df = pd.DataFrame()
feature_importance_df = pd.DataFrame()

# Training parameters
num_folds = 5
stratified = True
SEED = 1001
early_rounds = 50
importance_save = True
display_conf = True
display_imp = True
mongo_save = True

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

# Create dictionary to send to MongoDB alongside validation score
mongo_dict = params
mongo_dict['training_dataset'] = train_name
mongo_dict['early_stopping_rounds'] = early_rounds
mongo_dict['seed'] = SEED
mongo_dict['num_folds'] = num_folds
mongo_dict['stratified'] = stratified
mongo_dict['notes'] = 'Trying all tsfresh features'

# Create y_true for scoring
target_df = train_df[['object_id', 'target']]
target_df = pd.concat([target_df, pd.get_dummies(target_df.target)], axis=1)
target_df = target_df.drop('target', axis=1)
y_true = target_df.iloc[:, 1:].values

del target_df

# MongoDB parameters
client = pymongo.MongoClient('mongodb://localhost:27017')
db = client.reporting
collection = db.validation

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

def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(0, figsize=(12, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    # plt.show()
    plt.savefig('output/lgb_importances.png')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    # plt.show()
    plt.savefig('output/lgb_confusion.png')

def smoteAdataset(Xig_train, yig_train, Xig_test, yig_test):
    
        
    sm=SMOTE(random_state=SEED)
    Xig_train_res, yig_train_res = sm.fit_sample(Xig_train, yig_train.ravel())

        
    return Xig_train_res, pd.Series(yig_train_res), Xig_test, pd.Series(yig_test)

with timer("Run LightGBM with kfold"):
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=SEED)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=SEED)

    # Create arrays and dataframes to store results
    if params['num_class'] == 14:
        oof_preds = np.zeros((train_df.shape[0], len(le.classes_)))
    elif params['num_class'] == 15:
        oof_preds = np.zeros((train_df.shape[0], len(le.classes_) + 1))

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
        
        oof_preds[valid_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)


    fold_importance_df["feature"] = feats
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    del clf

    score = multi_weighted_logloss(y_true, oof_preds)
    print('Log-loss on oof preds: {}'.format(score))
    print("__________")

    mongo_dict['score'] = score

    if mongo_save == True:
        collection.insert_one(mongo_dict)

    if display_imp == True:
        display_importances(feature_importance_df)

    # Save feature importance df as csv
    if importance_save == True:
        feature_importance_df = feature_importance_df.groupby('feature').agg('mean').drop('fold', axis = 1).sort_values('importance')
        feature_importance_df.to_csv('output/importance.csv')

if display_conf == True:
    unique_y = np.unique(train_df['target'])
    class_map = dict()
    for i,val in enumerate(unique_y):
        class_map[val] = i
            
    y_map = np.zeros((train_df['target'].shape[0],))
    y_map = np.array([class_map[val] for val in train_df['target']])

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_map, np.argmax(oof_preds,axis=-1))
    np.set_printoptions(precision=2)

    sample_sub = pd.read_csv('data/sample_submission.csv')
    class_names = list(sample_sub.columns[1:-1])
    del sample_sub;gc.collect()

    # Plot non-normalized confusion matrix
    plt.figure(1, figsize=(10,10))
    foo = plot_confusion_matrix(cnf_matrix, classes=class_names,normalize=True,
                        title='Confusion matrix')
