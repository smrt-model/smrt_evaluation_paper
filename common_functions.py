'''Contains functions that may be called for ASMEx, PAMIR and/or NoSREx analysis'''
import pandas as pd
import numpy as np
from scipy.stats import linregress

from smrt.core.globalconstants import DENSITY_OF_ICE

def symmetrize_microstructure(file):
    '''Takes in micro-CT data analyzed in three dimensions, returns dataframe of microstructure parameters needed for SMRT
    
    Format of input file is specific to this study. Length scales are in mm (need to be in m for SMRT)
    Parameters analyzed separately along cartesian co-ordinates and averaged over 3-D

    
    ''' 
    
    df = pd.read_csv(file)
    if df[' 4:l_c_y'][0] == ' NaN':
        # These are PAMIR data (no analysis in y direction)
        density = df[' 2:phi_2'].mul(DENSITY_OF_ICE)  # Density
        d_shs = df[' 24:d_shs'].mul(1e-3)  # Change units to m
        tau = df[' 25:tau'] # Unitless
        # y-component assumed to be same as x-component
        # Take mean, and change units to m
        l_c = (df[' 3:l_c_x'].mul(2) + df[' 5:l_c_z']).mul(1e-3 / 3)
        l_ex = (df[' 9:l_ex_x'].mul(2) + df[' 11:l_ex_z']).mul(1e-3 / 3)  
        d_sph = (df[' 6:d_sph_x'].mul(2) + df[' 8:d_sph_z']).mul(1e-3 / 3)  
        xi_ts = (df[' 12:xi_ts_x'].mul(2) + df[' 14:xi_ts_z']).mul(1e-3 / 3)  
        domain_ts = (df[' 15:d_ts_x'].mul(2) + df[' 17:d_ts_z']).mul(1e-3 / 3)  
        xi_grf = (df[' 18:xi_grf_x'].mul(2) + df[' 20:xi_grf_z']).mul(1e-3 / 3)   
        domain_grf = (df[' 21:d_grf_x'].mul(2) + df[' 23:d_grf_z']).mul(1e-3 / 3)  
    else:
        # These are ASMEx or NoSREx data
        density = df[' 2:phi_2'].mul(DENSITY_OF_ICE)
        # Microstructure params
        d_shs = df[' 24:d_shs'].mul(1e-3)  # Change units to m
        tau = df[' 25:tau']  # Unitless
        l_c = (df[' 3:l_c_x'] + df[' 4:l_c_y'] + df[' 5:l_c_z']).mul(1e-3 / 3)  
        l_ex = (df[' 9:l_ex_x'] + df[' 10:l_ex_y'] + df[' 11:l_ex_z']).mul(1e-3 / 3)  
        d_sph = (df[' 6:d_sph_x'] + df[' 7:d_sph_y'] + df[' 8:d_sph_z']).mul(1e-3 / 3) 
        xi_ts = (df[' 12:xi_ts_x'] + df[' 13:xi_ts_y'] + df[' 14:xi_ts_z']).mul(1e-3 / 3) 
        domain_ts = (df[' 15:d_ts_x'] + df[' 16:d_ts_y'] + df[' 17:d_ts_z']).mul(1e-3 / 3)  
        xi_grf = (df[' 18:xi_grf_x'] + df[' 19:xi_grf_y'] + df[' 20:xi_grf_z']).mul(1e-3 / 3)  
        domain_grf = (df[' 21:d_grf_x'] + df[' 22:d_grf_y'] + df[' 23:d_grf_z']).mul(1e-3 / 3)  
        
    # Make output dataframe
    frame={'density':density, 'd_shs':d_shs, 'tau':tau, 'l_c':l_c, 'l_ex':l_ex, 'd_sph':d_sph,
          'xi_ts':xi_ts, 'domain_ts':domain_ts, 'xi_grf':xi_grf, 'domain_grf':domain_grf}
    return pd.DataFrame(frame)


def iqr(df):
    '''
    Calculates interquartile range for pandas dataframe
    
    '''
    return df.quantile(0.75) - df.quantile(0.25)


def microstructure_table_format(ds):
    '''
    Takes data series of microstructure parameter statistics, changes m units to mm for paper
    
    '''
    non_scaled = ds.loc[['density', 'tau']]
    scaled = ds.loc[['d_shs', 'l_ex', 'd_sph', 'xi_ts', 'domain_ts', 'xi_grf', 'domain_grf']] * 1e3
    # Concat into new data series
    newds = np.round(pd.concat([non_scaled, scaled]), 2)
    return newds.reindex(['density', 'd_shs', 'tau', 'l_ex', 'd_sph', 'xi_ts', 'domain_ts', 'xi_grf', 'domain_grf'])


# Calculate RMSE, ME
def rmse(obs, sims):
    '''
    Returns root mean squared error between observations and simulations
    '''
    return np.sqrt(((obs - sims) ** 2).mean().values)


def me(obs, sims):
    '''
    Returns mean error between observations and sims. Negative error means simulations underestimated
    '''
    return (sims-obs).mean().values
    

def regression_coefficient(obs, sims):
    '''
    Returns gradient and intercept for best fit line, and r^2 value
    '''
    x = obs.values.flatten()
    y = sims.values.flatten()
    x_where_obs = x[~np.isnan(x)]
    y_where_obs = y[~np.isnan(x)]
    gradient, intercept, r_value, p_value, std_err = linregress(x_where_obs,y_where_obs)
    return gradient, intercept, r_value**2


def microstructure_colour_list():
    return {'Exponential': 'r', 'IndependentSphere':'b', 'StickyHardSpheres':'c', 'TeubnerStrey':'m', 'GaussianRandomField':'purple'}


def microstructure_short_labels():    
    return {'Exponential': 'EXP', 'IndependentSphere':'IND', 'StickyHardSpheres':'SHS', 'TeubnerStrey':'TS', 'GaussianRandomField':'GRF'}
