import numpy as np
import pandas as pd

def data_process(df, test=False):
    # Feature engineering
    df['uncertainty'] = (100*df['flux_err'])/abs(df['flux'])
    df['flux_diff']=df['flux']-df['flux'].mean()
    df['flux_diff_2']=(df['flux']-df['flux'].mean())**2

    # Aggregate time-series features
    aggregate = {
    'mjd': ['min', 'max', 'mean', 'var'],
    'flux': ['min', 'max', 'mean', 'var'],
    'flux_err': ['min', 'max', 'mean', 'var', 'sum'],
    'detected': ['mean', 'var'],
    'uncertainty': ['mean', 'var', 'min', 'max', 'sum'],
    'flux_diff': ['mean', 'var', 'min', 'max', 'sum'],
    'flux_diff_2': ['mean', 'var', 'min', 'max', 'sum'],
    }

    # Aggregate both datasets
    df = df.groupby(['object_id', 'passband'], as_index=False).agg(aggregate)

    # Change from multi-level column names to single-level
    df.columns = pd.Index([e[0] + "_" + e[1] for e in df.columns.tolist()])

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

    if not test:
        train_meta = pd.read_csv('../../data/training_set_metadata.csv', index_col=False)
        df = train_meta.merge(df, on='object_id')

    return df