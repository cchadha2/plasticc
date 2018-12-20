import numpy as np
import pandas as pd

from tsfresh.convenience.relevant_extraction import extract_relevant_features
from tsfresh.feature_extraction import extract_features
from astropy.cosmology import FlatLambdaCDM

def data_process(df, test=False):
   
    fcp = {
        
        'mjd': {
            'maximum': None, 
            'minimum': None,
            'mean_change': None,
            'mean_abs_change': None,
        },
    }

    if not test:
        target = pd.read_csv('data/training_set_metadata.csv', squeeze=True,  usecols=['target'])
    else:
        target = pd.read_csv('data/test_set_metadata.csv', squeeze=True,  usecols=['target'])

    print('target: {}'.format(target.shape))

    extracted_features = extract_relevant_features(df,
                                                   y=target,
                                                   column_id='object_id',
                                                   column_sort='mjd',
                                                   column_kind='passband',
                                                   column_value='flux',
                                                   n_jobs=4)

    df_det = df[df['detected']==1].copy()
    agg_df_mjd = extract_features(df_det, column_id='object_id', column_value='mjd', 
                                  default_fc_parameters=fcp['mjd'], n_jobs=4)
    agg_df_mjd['mjd_diff_det'] = agg_df_mjd['mjd__maximum'].values - agg_df_mjd['mjd__minimum'].values
    del agg_df_mjd['mjd__maximum'], agg_df_mjd['mjd__minimum']

    agg_df_mjd.index.rename('object_id', inplace=True)      
    agg_df_ts = pd.concat([extracted_features,
                           agg_df_mjd], axis=1).reset_index()

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

    if not test:
        train_meta = pd.read_csv('data/training_set_metadata.csv', index_col=False)
        train_meta['distmod'] = cosmo.distmod(train_meta['hostgal_photoz'])
        train_meta = train_meta.drop(['hostgal_specz','ddf','ra','decl','gal_l','gal_b'], axis=1)
        train_meta['hostgal_photoz_uncertainty'] = (100*train_meta['hostgal_photoz_err'])/abs(train_meta['hostgal_photoz'])
        train_meta['hostgal_photoz_ratio_sq'] = np.power(train_meta['hostgal_photoz'].values / train_meta['hostgal_photoz_err'].values, 2.0)
        train_meta['hostgal_photoz_by_hostgal_photoz_ratio_sq'] = train_meta['hostgal_photoz'].values * train_meta['hostgal_photoz_ratio_sq'].values
        df = train_meta.merge(agg_df_ts, on='object_id')
    else:
        test_meta = pd.read_csv('data/test_set_metadata.csv', index_col=False)
        test_meta['distmod'] = cosmo.distmod(test_meta['hostgal_photoz'])
        test_meta = test_meta.drop(['hostgal_specz','ddf','ra','decl','gal_l','gal_b'], axis=1)
        test_meta['hostgal_photoz_uncertainty'] = (100*test_meta['hostgal_photoz_err'])/abs(test_meta['hostgal_photoz'])
        test_meta['hostgal_photoz_ratio_sq'] = np.power(test_meta['hostgal_photoz'].values / test_meta['hostgal_photoz_err'].values, 2.0)
        test_meta['hostgal_photoz_by_hostgal_photoz_ratio_sq'] = test_meta['hostgal_photoz'].values * test_meta['hostgal_photoz_ratio_sq'].values
        df = test_meta.merge(agg_df_ts, on='object_id')
    
    return df