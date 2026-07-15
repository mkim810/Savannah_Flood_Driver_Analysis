import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
from tqdm import tqdm

# --- Configuration ---
data_dir   = r"E:\Draft\Technical_Paper\Flood_Sensitivity\paper2_sensitivity"
xgb_dir    = r"E:\Draft\Technical_Paper\Flood_Sensitivity\paper2_sensitivity"
output_dir = r"E:\Draft\Technical_Paper\Flood_Sensitivity\paper2_sensitivity"

input_path        = os.path.join(data_dir,   "sensitivity_results.csv")
xgb_r2_path       = os.path.join(xgb_dir,   "r2_per_node.csv")
xgb_residual_path = os.path.join(xgb_dir,   "residuals.csv")

n_jobs   = -1  # Use all available CPU cores
N_COMP   = 3   # PLSR components

FEATURES = ['Downstream_Factor', 'Precipitation_Factor', 'Upstream_Factor']
TARGET   = 'Max_Depth'

# --- Per-Node RF + PLSR Function ---
def run_models_for_node(node_id, group):
    """
    Fits Random Forest and PLSR on all simulations for a single node.
    Returns R2 results and residual rows.
    """
    X = group[FEATURES].values
    y = group[TARGET].values

    # --- Random Forest ---
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=1           # Keep at 1; parallelism is across nodes
    )
    rf_model.fit(X, y)
    y_pred_rf = rf_model.predict(X)
    r2_rf     = r2_score(y, y_pred_rf)

    # --- PLSR ---
    pls_model = PLSRegression(n_components=N_COMP)
    pls_model.fit(X, y)
    y_pred_pls = pls_model.predict(X).flatten()
    r2_pls     = r2_score(y, y_pred_pls)

    # --- R2 row (one per node) ---
    r2_row = {
        'Node_ID'          : node_id,
        'R2_RandomForest'  : r2_rf,
        'R2_PLSR'          : r2_pls
    }

    # --- Residual rows (one per simulation) ---
    residual_df = group[['Simulation_ID', 'Node_ID'] + FEATURES + [TARGET]].copy()
    residual_df['Predicted_RF']      = y_pred_rf
    residual_df['Residual_RF']       = y_pred_rf   - y
    residual_df['Predicted_PLSR']    = y_pred_pls
    residual_df['Residual_PLSR']     = y_pred_pls  - y

    return r2_row, residual_df


# --- Load Data ---
print("Loading sensitivity results...")
df = pd.read_csv(input_path)
print(f"  Total rows    : {len(df):,}")
print(f"  Unique nodes  : {df['Node_ID'].nunique():,}")

node_groups = [(node_id, group.reset_index(drop=True)) for node_id, group in df.groupby('Node_ID')]
print(f"  Nodes to process: {len(node_groups):,}")

# --- Parallel Execution ---
print(f"\nRunning Random Forest & PLSR in parallel (n_jobs={n_jobs})...")
results = Parallel(n_jobs=n_jobs, backend='loky')(
    delayed(run_models_for_node)(node_id, group)
    for node_id, group in tqdm(node_groups)
)

# --- Unpack Results ---
print("\nAggregating results...")
r2_rows      = [r[0] for r in results]
residual_dfs = [r[1] for r in results]

r2_new_df       = pd.DataFrame(r2_rows)
residual_new_df = pd.concat(residual_dfs, ignore_index=True)

# --- Load XGBoost Results ---
print("Loading XGBoost results...")
xgb_r2_df       = pd.read_csv(xgb_r2_path).rename(columns={'R2': 'R2_XGBoost'})
xgb_residual_df = pd.read_csv(xgb_residual_path).rename(columns={
    'Predicted_Depth' : 'Predicted_XGBoost',
    'Residual'        : 'Residual_XGBoost'
})

# --- Merge R2 ---
r2_final = xgb_r2_df.merge(r2_new_df, on='Node_ID', how='inner') \
                     .sort_values('Node_ID') \
                     .reset_index(drop=True)

r2_final = r2_final[['Node_ID', 'R2_XGBoost', 'R2_RandomForest', 'R2_PLSR']]

# --- Merge Residuals ---
residual_final = xgb_residual_df.merge(
    residual_new_df[['Simulation_ID', 'Node_ID', 'Predicted_RF', 'Residual_RF', 'Predicted_PLSR', 'Residual_PLSR']],
    on=['Simulation_ID', 'Node_ID'],
    how='inner'
).sort_values(['Node_ID', 'Simulation_ID']).reset_index(drop=True)

residual_final = residual_final[[
    'Simulation_ID', 'Node_ID',
    'Downstream_Factor', 'Precipitation_Factor', 'Upstream_Factor',
    'Max_Depth',
    'Predicted_XGBoost', 'Residual_XGBoost',
    'Predicted_RF',      'Residual_RF',
    'Predicted_PLSR',    'Residual_PLSR'
]]

# --- Save ---
r2_out_path       = os.path.join(output_dir, "r2_per_node_all.csv")
residual_out_path = os.path.join(output_dir, "residuals_all.csv")

r2_final.to_csv(r2_out_path, index=False)
residual_final.to_csv(residual_out_path, index=False)

print(f"\nSaved R2 results  -> {r2_out_path}")
print(f"Saved Residuals   -> {residual_out_path}")
print(f"\nR2 Summary:\n{r2_final[['R2_XGBoost','R2_RandomForest','R2_PLSR']].describe()}")
print("Done.")
