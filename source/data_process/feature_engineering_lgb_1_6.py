import numpy as np
import pandas as pd

from tsfresh.feature_extraction import extract_features
from astropy.cosmology import FlatLambdaCDM

def data_process(df, test=False):
    # Feature engineering
    df['flux_err_perc'] = (100*df['flux_err'])/abs(df['flux'])
    df['flux_diff']=df['flux']-df['flux'].mean()
    df['flux_diff_sq']=(df['flux']-df['flux'].mean())**2
    df['flux_ratio_sq'] = np.power(df['flux'].values / df['flux_err'].values, 2.0)
    df['flux_by_flux_ratio_sq'] = df['flux'].values * df['flux_ratio_sq'].values
    df['magnitude'] = (-2.5)*np.log10(df['flux'].values)
    # df['flux_upper'] = df['flux'] + df['flux_err']
    # df['flux_lower'] = df['flux'] - df['flux_err']

    # Aggregate time-series features
    aggregate = {
    'flux': ['min', 'max', 'mean', 'median', 'skew', 'std'],
    'flux_err': ['min', 'max', 'mean'],
    'detected': ['mean'],
    'flux_err_perc': ['mean', 'min', 'max', 'sum', 'median', 'skew', 'std'],
    'flux_diff': ['mean', 'var', 'min', 'max', 'sum'],
    'flux_diff_sq': ['mean', 'var', 'min', 'max', 'sum'],
    'flux_ratio_sq':['sum','skew'],
    'flux_by_flux_ratio_sq':['sum','skew'],
    # 'magnitude': ['min', 'max', 'mean', 'var', 'median', 'skew', 'std'],
    # 'flux_upper': ['mean'],
    # 'flux_lower': ['mean'],
    }

    # Aggregate dataset
    agg_df = df.groupby(['object_id']).agg(aggregate)

    # Change from multi-level column names to single-level
    agg_df.columns = [ '{}_{}'.format(k, agg) for k in aggregate.keys() for agg in aggregate[k]]

    agg_df['flux_w_mean'] = agg_df['flux_by_flux_ratio_sq_sum'].values / agg_df['flux_ratio_sq_sum'].values
    agg_df['flux_max_min'] = agg_df['flux_max'].values - agg_df['flux_min'].values
    agg_df['flux_max_min_ave'] = agg_df['flux_max_min'].values / agg_df['flux_mean'].values

    fcp = {
        'flux': {
            'longest_strike_above_mean': None,
            'longest_strike_below_mean': None,
            'mean_change': None,
            'mean_abs_change': None,
            'length': None,
        },
                
        'flux_by_flux_ratio_sq': {
            'longest_strike_above_mean': None,
            'longest_strike_below_mean': None,       
        },
                
        'flux_passband': {
            'fft_coefficient': [
                    {'coeff': 0, 'attr': 'abs'}, 
                    {'coeff': 1, 'attr': 'abs'}
                ],
            'kurtosis' : None, 
            'skewness' : None,
        },
                
        'mjd': {
            'maximum': None, 
            'minimum': None,
            'mean_change': None,
            'mean_abs_change': None,
        },
    }

    # Add more features with
    agg_df_ts_flux_passband = extract_features(df, 
                                               column_id='object_id', 
                                               column_sort='mjd', 
                                               column_kind='passband', 
                                               column_value='flux', 
                                               default_fc_parameters=fcp['flux_passband'], n_jobs=4)

    agg_df_ts_flux = extract_features(df, 
                                      column_id='object_id', 
                                      column_value='flux', 
                                      default_fc_parameters=fcp['flux'], n_jobs=4)

    agg_df_ts_flux_by_flux_ratio_sq = extract_features(df, 
                                      column_id='object_id', 
                                      column_value='flux_by_flux_ratio_sq', 
                                      default_fc_parameters=fcp['flux_by_flux_ratio_sq'], n_jobs=4)

    # Add smart feature that is suggested here https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696#410538
    # dt[detected==1, mjd_diff:=max(mjd)-min(mjd), by=object_id]
    df_det = df[df['detected']==1].copy()
    agg_df_mjd = extract_features(df_det, column_id='object_id', column_value='mjd', 
                                  default_fc_parameters=fcp['mjd'], n_jobs=4)
    agg_df_mjd['mjd_diff_det'] = agg_df_mjd['mjd__maximum'].values - agg_df_mjd['mjd__minimum'].values
    del agg_df_mjd['mjd__maximum'], agg_df_mjd['mjd__minimum']

    agg_df_ts_flux_passband.index.rename('object_id', inplace=True) 
    agg_df_ts_flux.index.rename('object_id', inplace=True) 
    agg_df_ts_flux_by_flux_ratio_sq.index.rename('object_id', inplace=True) 
    agg_df_mjd.index.rename('object_id', inplace=True)      
    agg_df_ts = pd.concat([agg_df, 
                           agg_df_ts_flux_passband, 
                           agg_df_ts_flux, 
                           agg_df_ts_flux_by_flux_ratio_sq, 
                           agg_df_mjd], axis=1).reset_index()
    
    # agg_df_ts = pd.merge(agg_df_ts, agg_df_mjd, on='id')
    # tsfresh returns a dataframe with an index name='id'
    # agg_df_ts.index.rename('object_id', inplace=True)
    
    # agg_df = pd.merge(agg_df, agg_df_ts, on='object_id')

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

    # df['abs_magnitude_min'] = df['magnitude_min'] - df['distmod']
    # df['abs_magnitude_max'] = df['magnitude_max'] - df['distmod']
    # df['abs_magnitude_mean'] = df['magnitude_mean'] - df['distmod']
    # df['abs_magnitude_median'] = df['magnitude_median'] - df['distmod']
    
    return df