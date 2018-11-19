import numpy as np
import pandas as pd

def remove_on_importance(df, remove_by_threshold = True, remove_by_rank = False):
    
    # Remove all features that weren't split on above some threshold times in most recent run
    importance = pd.read_csv('output/importance.csv')

    if remove_by_threshold == True:
        importance_threshold = 2
        importance_red = importance[importance['importance']<=importance_threshold]
        importance_red.reset_index(inplace = True)
        importance_list = list(importance_red['feature'])
        df = df.drop(importance_list, axis = 1)

        del importance_red
        del importance_list

    if remove_by_rank == True:
        importance_rank = 650
        importance_red = importance.sort_values('importance', ascending = False)[importance_rank:]
        importance_red.reset_index(inplace = True)
        importance_list = list(importance_red['feature'])
        df = df.drop(importance_list, axis = 1)

        del importance_red
        del importance_list
    
    del importance

    return df

def data_process(df, test=False, importance_prune=False):
    # Feature engineering
    df['flux_uncertainty'] = (100*df['flux_err'])/abs(df['flux'])
    df['flux_diff']=df['flux']-df['flux'].mean()
    df['flux_diff_sq']=(df['flux']-df['flux'].mean())**2
    df['flux_ratio_sq'] = np.power(df['flux'].values / df['flux_err'].values, 2.0)
    df['flux_by_flux_ratio_sq'] = df['flux'].values * df['flux_ratio_sq'].values

    # Aggregate time-series features
    aggregate = {
    'mjd': ['min', 'max', 'mean', 'var'],
    'flux': ['min', 'max', 'mean', 'var'],
    'flux_err': ['min', 'max', 'mean', 'var', 'sum'],
    'detected': ['mean', 'var'],
    'flux_uncertainty': ['mean', 'var', 'min', 'max', 'sum'],
    'flux_diff': ['mean', 'var', 'min', 'max', 'sum'],
    'flux_diff_sq': ['mean', 'var', 'min', 'max', 'sum'],
    'flux_ratio_sq':['sum','skew'],
    'flux_by_flux_ratio_sq':['sum','skew'],
    }

    # Aggregate both datasets
    df = df.groupby(['object_id', 'passband'], as_index=False).agg(aggregate)

    # Change from multi-level column names to single-level
    df.columns = pd.Index([e[0] + "_" + e[1] for e in df.columns.tolist()])

    df['flux_w_mean'] = df['flux_by_flux_ratio_sq_sum'].values / df['flux_ratio_sq_sum'].values
    df['flux_max_min'] = df['flux_max'].values - df['flux_min'].values
    df['flux_max_min_ave'] = df['flux_max_min'].values / df['flux_mean'].values
    # df['flux_max_min_w_mean'] = df['flux_max_min'].values / df['flux_w_mean'].values

    # Rename two amended column names in above
    df = df.rename(columns={'object_id_': 'object_id', 'passband_': 'passband'})

    # Create a list of columns to drop from final datasets
    original_columns=list(df.columns)
    original_columns.remove('object_id')

    # Create new features for each passband
    for n in range(2,len(df.columns)):
        df[str(df.columns[1]) + '_0_' + str(df.columns[n])]=df[df['passband']==0][df.columns[n]]
        df[str(df.columns[1]) + '_1_' + str(df.columns[n])]=df[df['passband']==1][df.columns[n]]
        df[str(df.columns[1]) + '_2_' + str(df.columns[n])]=df[df['passband']==2][df.columns[n]]
        df[str(df.columns[1]) + '_3_' + str(df.columns[n])]=df[df['passband']==3][df.columns[n]]
        df[str(df.columns[1]) + '_4_' + str(df.columns[n])]=df[df['passband']==4][df.columns[n]]
        df[str(df.columns[1]) + '_5_' + str(df.columns[n])]=df[df['passband']==5][df.columns[n]]

    # Sum all features to get rid of sparse data
    df = df.groupby('object_id', as_index=False).agg(sum)

    df = df.drop(original_columns, axis=1)

    if importance_prune == True:
        with timer("Post-processing"):
            df = remove_on_importance(df)
            print("df shape:", df.shape)

    if not test:
        train_meta = pd.read_csv('data/training_set_metadata.csv', index_col=False)
        train_meta = train_meta.drop('hostgal_specz', axis=1)
        df = train_meta.merge(df, on='object_id')

    return df