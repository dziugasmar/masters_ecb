#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels import PanelOLS
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor
warnings.filterwarnings('ignore')

# Load data 
df = pd.read_excel("/Users/dziugas/Desktop/Master's/Datasets/main.xlsx")

# Convert "#N/A" and "NULL" strings to actual NaN values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].replace(['#N/A', 'NULL'], np.nan)
        # Try to convert numeric columns
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except:
            pass

# Data preparation
df['Date'] = pd.to_datetime(df['Date'])
print(f"Initial data shape: {df.shape}")

# IMPORTANT: Mark days with tone shocks
df['has_tone'] = df['tone'].notna()

# Calculate delta variables - only after marking tone days
df['st_yield_delta'] = df.groupby('country')['st_yield'].transform('diff') * 100 * (-1)
df['mt_yield_delta'] = df.groupby('country')['mt_yield'].transform('diff') * 100 * (-1)
df['lt_yield_delta'] = df.groupby('country')['lt_yield'].transform('diff') * 100 * (-1)
df['ois1m_delta'] = df.groupby('country')['ois1m'].transform('diff') * (-1) * 100
df['ois1y_delta'] = df.groupby('country')['ois1y'].transform('diff') * (-1) * 100
df['ois5y_delta'] = df.groupby('country')['ois5y'].transform('diff') * (-1) * 100
df['fx_delta'] = df.groupby('country')['fx'].transform(lambda x: np.log(x) - np.log(x.shift(1))) * (-1)

# Custom equity_delta calculation
def compute_equity_delta(row, prev_close):
    if pd.notnull(row['equity_open']) and pd.notnull(row['equity_close']):
        return np.log(row['equity_close']) - np.log(row['equity_open'])
    elif pd.notnull(row['equity_close']) and pd.notnull(prev_close):
        return np.log(row['equity_close']) - np.log(prev_close)
    else:
        return np.nan

def get_equity_delta(group):
    prev_closes = group['equity_close'].shift(1)
    return group.apply(lambda row: compute_equity_delta(row, prev_closes.loc[row.name]), axis=1)

df['equity_delta'] = df.groupby('country', group_keys=False).apply(get_equity_delta)

# General delta calculation for open/close pairs
def compute_delta(row, open_col, close_col, prev_close):
    if pd.notnull(row[open_col]) and pd.notnull(row[close_col]):
        return np.log(row[close_col]) - np.log(row[open_col])
    elif pd.notnull(row[close_col]) and pd.notnull(prev_close):
        return np.log(row[close_col]) - np.log(prev_close)
    else:
        return np.nan

def get_delta(group, open_col, close_col, delta_col):
    prev_closes = group[close_col].shift(1)
    group[delta_col] = group.apply(lambda row: compute_delta(row, open_col, close_col, prev_closes.loc[row.name]), axis=1)
    return group

df = df.groupby('country', group_keys=False).apply(get_delta, 'v2x_open', 'v2x_close', 'v2x_delta')

# Country group dummies
df['Baltic'] = df['country'].isin(['LT', 'LV', 'EE']).astype(int)
df['Nordic'] = df['country'].isin(['FI', 'SE', 'NO', 'DK', 'IS']).astype(int)

# Create tone dummies (using Neutral as reference)
df['tone_dovish'] = (df['tone'] == 'Dovish').astype(int)
df['tone_hawkish'] = (df['tone'] == 'Hawkish').astype(int)
df['tone_neutral'] = (df['tone'] == 'Neutral').astype(int)
# Create event type dummies
df['is_press'] = (df['type'] == 'press').astype(int) if 'type' in df.columns else 0
df['is_speech'] = (df['type'] == 'speech').astype(int) if 'type' in df.columns else 0


# Create interactions
# Tone interactions with country groups
df['tone_dovish_Baltic'] = df['tone_dovish'] * df['Baltic']
df['tone_hawkish_Baltic'] = df['tone_hawkish'] * df['Baltic']
df['tone_dovish_Nordic'] = df['tone_dovish'] * df['Nordic']
df['tone_hawkish_Nordic'] = df['tone_hawkish'] * df['Nordic']
df['tone_neutral_Baltic'] = df['tone_neutral'] * df['Baltic']
df['tone_neutral_Nordic'] = df['tone_neutral'] * df['Nordic']

# Tone interactions with event type
df['tone_dovish_press'] = df['tone_dovish'] * df['is_press']
df['tone_hawkish_press'] = df['tone_hawkish'] * df['is_press']
df['tone_dovish_speech'] = df['tone_dovish'] * df['is_speech']
df['tone_hawkish_speech'] = df['tone_hawkish'] * df['is_speech']
df['tone_neutral_press'] = df['tone_neutral'] * df['is_press']
df['tone_neutral_speech'] = df['tone_neutral'] * df['is_speech']

# Three-way interactions (tone * country group * event type)
df['tone_dovish_Baltic_press'] = df['tone_dovish'] * df['Baltic'] * df['is_press']
df['tone_hawkish_Baltic_press'] = df['tone_hawkish'] * df['Baltic'] * df['is_press']
df['tone_dovish_Nordic_press'] = df['tone_dovish'] * df['Nordic'] * df['is_press']
df['tone_hawkish_Nordic_press'] = df['tone_hawkish'] * df['Nordic'] * df['is_press']
df['tone_neutral_Baltic_press'] = df['tone_neutral'] * df['Baltic'] * df['is_press']
df['tone_neutral_Nordic_press'] = df['tone_neutral'] * df['Nordic'] * df['is_press']

df['tone_dovish_Baltic_speech'] = df['tone_dovish'] * df['Baltic'] * df['is_speech']
df['tone_hawkish_Baltic_speech'] = df['tone_hawkish'] * df['Baltic'] * df['is_speech']
df['tone_dovish_Nordic_speech'] = df['tone_dovish'] * df['Nordic'] * df['is_speech']
df['tone_hawkish_Nordic_speech'] = df['tone_hawkish'] * df['Nordic'] * df['is_speech']
df['tone_neutral_Baltic_speech'] = df['tone_neutral'] * df['Baltic'] * df['is_speech']
df['tone_neutral_Nordic_speech'] = df['tone_neutral'] * df['Nordic'] * df['is_speech']

# Winsorize variables at 1%/99% quantiles
def winsorize_series(s, lower=0.01, upper=0.99):
    return s.clip(lower=s.quantile(lower), upper=s.quantile(upper))


# Replace infinite values with NaN
df['equity_delta'] = df['equity_delta'].replace([np.inf, -np.inf], np.nan)

# Print diagnostic information about missing values
print("\nMissing values in key variables:")
for col in ['st_yield_delta', 'mt_yield_delta', 'lt_yield_delta', 'equity_delta']:
    missing = df[col].isna().sum()
    total = len(df)
    print(f"{col}: {missing} missing out of {total} ({missing/total*100:.2f}%)")

# Filter to include only days with tone data
df = df[df['has_tone']].copy()
print(f"\nDays with tone shocks: {len(df)}")

# Create OIS regional interactions
df['ois1m_delta_Nordic'] = df['ois1m_delta'] * df['Nordic']
df['ois1m_delta_Baltic'] = df['ois1m_delta'] * df['Baltic']
df['ois1y_delta_Nordic'] = df['ois1y_delta'] * df['Nordic']
df['ois1y_delta_Baltic'] = df['ois1y_delta'] * df['Baltic']
df['ois5y_delta_Nordic'] = df['ois5y_delta'] * df['Nordic']
df['ois5y_delta_Baltic'] = df['ois5y_delta'] * df['Baltic']

# Variables for regression
# Updated control vars with regional OIS interactions
control_vars = [
    'ois1m_delta', 'ois1m_delta_Nordic', 'ois1m_delta_Baltic',
    'ois1y_delta', 'ois1y_delta_Nordic', 'ois1y_delta_Baltic',
    'ois5y_delta', 'ois5y_delta_Nordic', 'ois5y_delta_Baltic',
    'fx_delta', 'v2x_delta'
]

# Event type interactions
event_vars = ['tone_dovish_press', 'tone_hawkish_press', 
              'tone_dovish_speech', 'tone_hawkish_speech', 'tone_dovish_Baltic_press', 'tone_hawkish_Baltic_press',
              'tone_dovish_Nordic_press',
              'tone_hawkish_Nordic_press',
              'tone_dovish_Baltic_speech', 'tone_hawkish_Baltic_speech',
              'tone_dovish_Nordic_speech', 'tone_hawkish_Nordic_speech', 'tone_neutral_Baltic_press', 'tone_neutral_Nordic_press', 'tone_neutral_press']

# Prepare panel data for regression
df_reg = df.copy()
df_reg = df_reg.set_index(['country', 'Date'])

# Calculate VIF for regression variables to check for multicollinearity
print("\n===== VARIANCE INFLATION FACTORS (VIF) =====")
print("VIF > 10 indicates problematic multicollinearity")

# Create a dataframe for VIF calculation (reset index to get country and Date as columns)
vif_data = df_reg.reset_index()[event_vars + control_vars].dropna()

# Calculate VIF for each predictor variable
vif_results = pd.DataFrame()
vif_results['Variable'] = vif_data.columns
vif_results['VIF'] = [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]

# Sort by VIF and print results
vif_results = vif_results.sort_values('VIF', ascending=False)
print(vif_results)

# Calculate number of unique time periods for bandwidth parameter
num_time_periods = df_reg.index.get_level_values(1).nunique()
print(f"\nNumber of unique time periods (T): {num_time_periods}")

# Calculate optimal bandwidth using rule of thumb: 4*(T/100)^(2/9)
optimal_bandwidth = int(4 * (num_time_periods/100)**(2/9))
print(f"Optimal bandwidth for Driscoll-Kraay standard errors: {optimal_bandwidth}")

# Function to run regression with Driscoll-Kraay HAC standard errors
def run_panel_regression(dep_var, exog_vars, data, title):
    formula = f"{dep_var} ~ {' + '.join(exog_vars)} + EntityEffects"
    subset = data.dropna(subset=[dep_var] + exog_vars)
    
    try:
        model = PanelOLS.from_formula(formula, data=subset, check_rank=False)
        
        # Use kernel-based covariance estimator with optimal bandwidth for Driscoll-Kraay standard errors
        result = model.fit(cov_type='kernel', kernel='bartlett', bandwidth=optimal_bandwidth)
        
        print(f"\n{title} (N={len(subset)})")
        print("=" * len(title))
        print(f"Using Driscoll-Kraay HAC standard errors with bandwidth = {optimal_bandwidth}")
        
        # Get the full summary to display all coefficients including interactions
        summary = result.summary
        print(summary)
        
        return result
    except Exception as e:
        print(f"\nERROR in {title}: {e}")
        print(f"Observations: {len(subset)}")
        return None

# Run regressions with event type interactions and OIS regional interactions
print("\n===== EVENT TYPE MODEL WITH REGIONAL OIS INTERACTIONS =====")
print(f"Using Driscoll-Kraay HAC standard errors with bandwidth = {optimal_bandwidth}")
for dep_var in ['st_yield_delta', 'mt_yield_delta', 'lt_yield_delta', 'equity_delta']:
    run_panel_regression(dep_var, event_vars + control_vars, df_reg, 
                         f"{dep_var} - Event Type Model with Regional OIS")

# NOTE: Implementation of Driscoll-Kraay standard errors in Python:
# 1. The 'linearmodels' package implements DK standard errors via the kernel covariance estimator
# 2. The bandwidth parameter determines the number of lags to use in the autocorrelation correction
# 3. The optimal bandwidth is calculated using Newey-West's rule of thumb: 4*(T/100)^(2/9)
#    where T is the number of time periods in the panel 
# 4. We've added regional interactions for OIS controls to capture differential effects across regions 