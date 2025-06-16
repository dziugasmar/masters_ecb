#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels import PanelOLS
import statsmodels.api as sm
import warnings
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats
warnings.filterwarnings('ignore')

# Create directory for saving LP results if it doesn't exist
os.makedirs('LP_results', exist_ok=True)

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
    if len(s.dropna()) > 10:  # Only winsorize if we have enough data
        return s.clip(lower=s.quantile(lower), upper=s.quantile(upper))
    return s


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
# Updated control vars with regional OIS interactions to match event_regression.py
control_vars = [
    'ois1m_delta', 'ois1m_delta_Nordic', 'ois1m_delta_Baltic',
    'ois1y_delta', 'ois1y_delta_Nordic', 'ois1y_delta_Baltic',
    'ois5y_delta', 'ois5y_delta_Nordic', 'ois5y_delta_Baltic',
    'fx_delta', 'v2x_delta'
]

# Event type interactions - exactly match event_regression.py
event_vars = [
    'tone_dovish_press', 'tone_hawkish_press', 
    'tone_dovish_speech', 'tone_hawkish_speech', 
    'tone_dovish_Baltic_press', 'tone_hawkish_Baltic_press',
    'tone_dovish_Nordic_press', 'tone_hawkish_Nordic_press',
    'tone_dovish_Baltic_speech', 'tone_hawkish_Baltic_speech',
    'tone_dovish_Nordic_speech', 'tone_hawkish_Nordic_speech',
    'tone_neutral_Baltic_press', 'tone_neutral_Nordic_press', 'tone_neutral_press'
]

# Combined model variables
combined_vars = event_vars + control_vars

# Prepare panel data for regressions
pooled_df = df.copy()

# Set horizon for local projections
H = 10

# Calculate lead variables for local projections
print("\nCalculating lead variables for local projections...")
for var in ['st_yield_delta', 'mt_yield_delta', 'lt_yield_delta', 'equity_delta']:
    for h in range(1, H+1):
        pooled_df[f'{var}_h{h}'] = pooled_df.groupby('country')[var].shift(-h)

# Set up panel data for local projections
pooled_reg = pooled_df.set_index(['country', 'Date'])

# Calculate VIF for regression variables to check for multicollinearity
print("\n===== VARIANCE INFLATION FACTORS (VIF) =====")
print("VIF > 10 indicates problematic multicollinearity")

# Calculate VIF for a subset of variables to avoid excessive computation
vif_subset = combined_vars[:20]  # Use first 20 variables for VIF calculation to avoid memory issues
vif_data = pooled_df[vif_subset].dropna()

# Calculate VIF for each predictor variable
vif_results = pd.DataFrame()
vif_results['Variable'] = vif_data.columns
vif_results['VIF'] = [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]

# Sort by VIF and print results
vif_results = vif_results.sort_values('VIF', ascending=False)
print(vif_results)

# Calculate number of unique time periods for bandwidth parameter
pooled_time_periods = pooled_reg.index.get_level_values(1).nunique()

# Calculate optimal bandwidth using rule of thumb: 4*(T/100)^(2/9)
base_bandwidth = int(4 * (pooled_time_periods/100)**(2/9))
print(f"\nPooled time periods: {pooled_time_periods}, base bandwidth: {base_bandwidth}")

# Local projection function for combined model approach
def local_proj_panel_combined(dep_var_base, data, exog_vars, H=10, base_bandwidth=6):
    """
    Run local projections for panel data with Driscoll-Kraay standard errors.
    Uses dynamic bandwidth: bandwidth = max(base_bandwidth, h + 2)
    """
    betas = pd.DataFrame(index=exog_vars, columns=range(H+1), dtype=float)
    ses = pd.DataFrame(index=exog_vars, columns=range(H+1), dtype=float)
    t_stats = pd.DataFrame(index=exog_vars, columns=range(H+1), dtype=float)
    p_values = pd.DataFrame(index=exog_vars, columns=range(H+1), dtype=float)
    r2 = pd.Series(index=range(H+1), dtype=float)
    nobs = pd.Series(index=range(H+1), dtype=int)
    
    for h in range(H+1):
        # Dynamic bandwidth based on horizon: max(base_bandwidth, h + 2)
        bandwidth = max(base_bandwidth, h + 2)
        
        if h == 0:
            # For h=0, use current variable (same as event regression)
            yname = dep_var_base
        else:
            # For h>0, use the pre-calculated lead variable
            yname = f'{dep_var_base}_h{h}'
        
        formula = f'{yname} ~ {" + ".join(exog_vars)} + EntityEffects'
        subset = data.dropna(subset=[yname] + exog_vars)
        
        if len(subset) == 0:
            print(f"WARNING: No data available for {dep_var_base} at horizon {h}")
            continue
            
        try:
            model = PanelOLS.from_formula(formula, data=subset, drop_absorbed=True)
            results = model.fit(cov_type='kernel', kernel='bartlett', bandwidth=bandwidth)
            
            print(f"Horizon {h}, Bandwidth {bandwidth}, N={len(subset)}")
            
            # Store results
            for var in exog_vars:
                if var in results.params.index:
                    betas.loc[var, h] = results.params.loc[var]
                    ses.loc[var, h] = results.std_errors.loc[var]
                    t_stats.loc[var, h] = results.tstats.loc[var]
                    p_values.loc[var, h] = results.pvalues.loc[var]
            
            r2[h] = results.rsquared
            nobs[h] = results.nobs
            
            # For h=0, print summary to verify results match event regression
            if h == 0:
                print(f"\n{dep_var_base} - Day 0 regression (should match event regression)")
                print("=" * 50)
                print(results)
        
        except Exception as e:
            print(f"Error in local projection at horizon {h}: {e}")
    
    return {
        'beta': betas,
        'se': ses,
        't_stat': t_stats, 
        'p_value': p_values,
        'r2': r2,
        'nobs': nobs
    }

# Plotting function for IRFs
def plot_irfs(results_dict, var_name, tone_var, region=None):
    """Plot impulse response functions with confidence intervals"""
    h = np.arange(H+1)
    
    betas = results_dict['beta']
    ses = results_dict['se']
    
    if tone_var in betas.index:
        irf = betas.loc[tone_var].to_numpy(dtype=float)
        err = ses.loc[tone_var].to_numpy(dtype=float)
        
        plt.figure(figsize=(10, 6))
        plt.fill_between(h, irf - 1.96*err, irf + 1.96*err, alpha=.25, color='lightblue')
        plt.plot(h, irf, 'b-o', mfc='white', linewidth=2)
        plt.axhline(0, color='k', lw=.5)
        
        region_str = f" - {region}" if region else ""
        plt.title(f'IRF for {var_name}{region_str} - {tone_var}')
        plt.ylabel('Response (percentage points)')
        plt.xlabel('Trading days after shock')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        tone_name = tone_var.replace('_', '-')
        filename = f"LP_results/{var_name}_{tone_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filename}")
        plt.close()
    else:
        print(f"Variable {tone_var} not found in results for {var_name}")

# Run local projections for combined model
print("\n===== RUNNING LOCAL PROJECTIONS WITH COMBINED MODEL =====")

combined_lp_results = {}

# Run for the 4 main dependent variables only, matching event_regression.py
for dep_var in ['st_yield_delta', 'mt_yield_delta', 'lt_yield_delta', 'equity_delta']:
    print(f"\nRunning local projections for {dep_var} with combined model")
    try:
        results = local_proj_panel_combined(
            dep_var, 
            pooled_reg, 
            combined_vars, 
            H=H, 
            base_bandwidth=base_bandwidth
        )
        combined_lp_results[dep_var] = results
    except Exception as e:
        print(f"Error running local projections for {dep_var}: {e}")

# Plot IRFs for the combined model
print("\n===== PLOTTING IMPULSE RESPONSE FUNCTIONS =====")

for dep_var in ['st_yield_delta', 'mt_yield_delta', 'lt_yield_delta', 'equity_delta']:
    if dep_var in combined_lp_results:
        # Plot Baltic specific interactions
        baltic_vars = [var for var in event_vars if 'Baltic' in var]
        for tone_var in baltic_vars:
            plot_irfs(combined_lp_results[dep_var], dep_var, tone_var, region="Baltic")
            
        # Plot Nordic specific interactions
        nordic_vars = [var for var in event_vars if 'Nordic' in var]
        for tone_var in nordic_vars:
            plot_irfs(combined_lp_results[dep_var], dep_var, tone_var, region="Nordic")
            
        # Plot base tone effects (not region-specific)
        base_vars = [var for var in event_vars if 'Baltic' not in var and 'Nordic' not in var]
        for tone_var in base_vars:
            plot_irfs(combined_lp_results[dep_var], dep_var, tone_var, region="Base")
            
        # Plot OIS variables effects
        print(f"\n===== PLOTTING OIS EFFECTS ON {dep_var} =====")
        ois_vars = [var for var in control_vars if 'ois' in var]
        for ois_var in ois_vars:
            plot_irfs(combined_lp_results[dep_var], dep_var, ois_var, region="OIS")

# Create summary DataFrames
print("\n===== CREATING SUMMARY STATISTICS =====")

# Function to create summary DataFrame
def create_summary_df(results_dict, var_name, tone_vars):
    """Create a summary DataFrame with coefficients, standard errors, t-stats, p-values and significance"""
    summary_data = []
    
    # Add all variables - both tone and OIS
    all_vars = tone_vars + [var for var in control_vars if 'ois' in var]
    
    for var in all_vars:
        if var in results_dict['beta'].index:
            for h in range(H+1):
                if not np.isnan(results_dict['beta'].loc[var, h]):
                    summary_data.append({
                        'Variable': var_name,
                        'Tone': var,
                        'Horizon': h,
                        'Coefficient': results_dict['beta'].loc[var, h],
                        'Std Error': results_dict['se'].loc[var, h],
                        't-stat': results_dict['t_stat'].loc[var, h],
                        'p-value': results_dict['p_value'].loc[var, h],
                        'R-squared': results_dict['r2'][h] if h in results_dict['r2'] else np.nan,
                        'N': results_dict['nobs'][h] if h in results_dict['nobs'] else np.nan,
                        'Significance': '***' if results_dict['p_value'].loc[var, h] < 0.01 else 
                                      ('**' if results_dict['p_value'].loc[var, h] < 0.05 else 
                                      ('*' if results_dict['p_value'].loc[var, h] < 0.1 else ''))
                    })
    
    return pd.DataFrame(summary_data)

# Create summary DataFrames
all_summaries = []

# Combined model summaries
for dep_var in ['st_yield_delta', 'mt_yield_delta', 'lt_yield_delta', 'equity_delta']:
    if dep_var in combined_lp_results:
        summary = create_summary_df(combined_lp_results[dep_var], dep_var, event_vars)
        all_summaries.append(summary)

# Combine all summaries
if all_summaries:
    full_summary = pd.concat(all_summaries, ignore_index=True)

    # Save the summary to CSV
    full_summary.to_csv('LP_results/local_projection_results_summary.csv', index=False)
    print("\nFull results saved to 'LP_results/local_projection_results_summary.csv'")

    # Print summary for key horizons (0, 1, 5, 10)
    key_horizons = [0, 1, 5, 10]
    key_summary = full_summary[full_summary['Horizon'].isin(key_horizons)]

    # Save key horizons summary
    key_summary.to_csv('LP_results/local_projection_key_horizons.csv', index=False)
    print("Key horizons summary saved to 'LP_results/local_projection_key_horizons.csv'")

# Function to create LaTeX tables
def create_latex_table(df, var_name, region):
    """Create a LaTeX table for a specific variable and region"""
    # Filter data
    if region == "Base":
        filtered = df[(df['Variable'] == var_name) & (~df['Tone'].str.contains('Baltic|Nordic'))]
    else:
        filtered = df[(df['Variable'] == var_name) & (df['Tone'].str.contains(region))]
    
    # Create a pivot table for easier LaTeX formatting
    pivot = filtered.pivot_table(
        index=['Tone', 'Horizon'],
        values=['Coefficient', 'Std Error', 'Significance'],
        aggfunc='first'
    ).reset_index()
    
    # Generate LaTeX code
    latex = f"\\begin{{table}}[htbp]\n"
    latex += f"\\centering\n"
    latex += f"\\caption{{Local Projection Results for {var_name} - {region} Effects}}\n"
    latex += f"\\begin{{tabular}}{{lcccc}}\n"
    latex += f"\\hline\n"
    latex += f"Tone & Horizon & Coefficient & Std Error & Significance \\\\ \n"
    latex += f"\\hline\n"
    
    # Add rows
    for _, row in pivot.iterrows():
        tone = row['Tone'].replace('_', '\\_')
        horizon = row['Horizon']
        coef = f"{row['Coefficient']:.4f}"
        stderr = f"{row['Std Error']:.4f}"
        sig = row['Significance'] if not pd.isna(row['Significance']) else ''
        
        latex += f"{tone} & {horizon} & {coef} & {stderr} & {sig} \\\\ \n"
    
    latex += f"\\hline\n"
    latex += f"\\end{{tabular}}\n"
    latex += f"\\end{{table}}\n"
    
    return latex

# Generate LaTeX tables
print("\n===== GENERATING LATEX TABLES =====")

if all_summaries:
    for var in ['st_yield_delta', 'mt_yield_delta', 'lt_yield_delta', 'equity_delta']:
        # Tone effects by region
        for region in ['Base', 'Baltic', 'Nordic']:
            latex_table = create_latex_table(full_summary, var, region)
            
            # Save to file
            with open(f'LP_results/{var}_{region}_latex.tex', 'w') as f:
                f.write(latex_table)
            
            print(f"LaTeX table saved to 'LP_results/{var}_{region}_latex.tex'")
        
        # OIS effects
        ois_vars = [x for x in control_vars if 'ois' in x]
        ois_summary = full_summary[(full_summary['Variable'] == var) & 
                                  (full_summary['Tone'].isin(ois_vars))]
        
        if not ois_summary.empty:
            # Generate LaTeX table for OIS effects
            latex = f"\\begin{{table}}[htbp]\n"
            latex += f"\\centering\n"
            latex += f"\\caption{{OIS Effects on {var}}}\n"
            latex += f"\\begin{{tabular}}{{lcccc}}\n"
            latex += f"\\hline\n"
            latex += f"OIS Variable & Horizon & Coefficient & Std Error & Significance \\\\ \n"
            latex += f"\\hline\n"
            
            # Add rows
            for _, row in ois_summary.iterrows():
                ois = row['Tone'].replace('_', '\\_')
                horizon = row['Horizon']
                coef = f"{row['Coefficient']:.4f}"
                stderr = f"{row['Std Error']:.4f}"
                sig = row['Significance'] if not pd.isna(row['Significance']) else ''
                
                latex += f"{ois} & {horizon} & {coef} & {stderr} & {sig} \\\\ \n"
            
            latex += f"\\hline\n"
            latex += f"\\end{{tabular}}\n"
            latex += f"\\end{{table}}\n"
            
            # Save to file
            with open(f'LP_results/{var}_OIS_effects_latex.tex', 'w') as f:
                f.write(latex)
            
            print(f"LaTeX table saved to 'LP_results/{var}_OIS_effects_latex.tex'")

print("\n===== COMBINED LOCAL PROJECTION ANALYSIS COMPLETE =====")
print("Results, plots, and LaTeX tables saved to the 'LP_results' folder") 