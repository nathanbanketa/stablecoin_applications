"""
PREDICTIVE MODELING FOR STABLECOIN ADOPTION
Machine Learning Approach (Decision Trees, Random Forests)
For: CSCI4911 Stablecoin Research Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PREDICTIVE MODELING: STABLECOIN ADOPTION")
print("Machine Learning Approach")
print("="*80)
print()

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================

print("STEP 1: Loading and preparing data...")
print("-"*80)

df = pd.read_csv("merged_dataset_for_regression.csv")

print(f"✓ Loaded data: {df.shape[0]} observations")
print("\nColumns available:")
print(df.columns.tolist())
print()

# Drop rows with missing dependent variable (adoption rate)
df_clean = df.dropna(subset=['Stablecoin_Share_Pct'])

print(f"After removing missing DV: {df_clean.shape[0]} observations")
print()

# ============================================================================
# STEP 2: SELECT FEATURES AND TARGET
# ============================================================================

print("STEP 2: Selecting features for modeling...")
print("-"*80)

# Target variable (what we're predicting)
target = 'Stablecoin_Share_Pct'

# Feature variables (what we're using to predict)
# Select only numeric columns and exclude identifiers
feature_cols = ['inflation_rate', 'internet_users_pct', 
                'mobile_subscriptions_per100', 'gdp_growth_pct',
                'exchange_rate_lcu_per_usd', 'remittances_pct_gdp']

# Drop rows with any missing values in features
df_model = df_clean[feature_cols + [target]].dropna()

print(f"Data for modeling: {df_model.shape[0]} observations × {len(feature_cols)} features")
print(f"\nTarget variable: {target}")
print(f"Feature variables:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i}. {col}")
print()

# ============================================================================
# STEP 3: PREPARE TRAIN/TEST SPLIT
# ============================================================================

print("STEP 3: Splitting data into train and test sets...")
print("-"*80)

X = df_model[feature_cols]
y = df_model[target]

print(f"Total samples: {len(X)}")
print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
print()

# For small datasets, use 60/40 or 70/30 split
if len(X) <= 10:
    test_size = 0.4
    print(f"Small dataset detected. Using 60/40 train/test split")
else:
    test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

print(f"Training set: {len(X_train)} observations")
print(f"Test set: {len(X_test)} observations")
print()

# ============================================================================
# STEP 4: DESCRIPTIVE STATISTICS
# ============================================================================

print("STEP 4: Feature statistics...")
print("-"*80)

print("\nFeature Summary Statistics:")
print(X_train.describe().to_string())
print()

print("Target (Stablecoin Share %) Statistics:")
print(f"  Mean: {y_train.mean():.2f}%")
print(f"  Std Dev: {y_train.std():.2f}%")
print(f"  Min: {y_train.min():.2f}%")
print(f"  Max: {y_train.max():.2f}%")
print()

# ============================================================================
# STEP 5: DECISION TREE REGRESSOR
# ============================================================================

print("="*80)
print("STEP 5: Training Decision Tree Model...")
print("="*80)
print()

# Decision tree is good for small datasets and interpretability
dt_model = DecisionTreeRegressor(max_depth=2, random_state=42, min_samples_leaf=1)
dt_model.fit(X_train, y_train)

# Make predictions
y_train_pred_dt = dt_model.predict(X_train)
y_test_pred_dt = dt_model.predict(X_test)

# Evaluate
dt_train_r2 = r2_score(y_train, y_train_pred_dt)
dt_test_r2 = r2_score(y_test, y_test_pred_dt)
dt_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_dt))
dt_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_dt))
dt_train_mae = mean_absolute_error(y_train, y_train_pred_dt)
dt_test_mae = mean_absolute_error(y_test, y_test_pred_dt)

print("DECISION TREE RESULTS:")
print("-"*80)
print(f"\nTraining Performance:")
print(f"  R² Score: {dt_train_r2:.4f}")
print(f"  RMSE: {dt_train_rmse:.2f} percentage points")
print(f"  MAE: {dt_train_mae:.2f} percentage points")
print(f"\nTest Performance:")
print(f"  R² Score: {dt_test_r2:.4f}")
print(f"  RMSE: {dt_test_rmse:.2f} percentage points")
print(f"  MAE: {dt_test_mae:.2f} percentage points")
print()

# Feature importance
dt_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Feature Importance (Decision Tree):")
print(dt_importance.to_string(index=False))
print()

# Visualize tree
fig, ax = plt.subplots(figsize=(16, 10))
plot_tree(dt_model, feature_names=feature_cols, filled=True, ax=ax, 
         rounded=True, fontsize=10)
plt.title('Decision Tree: Stablecoin Adoption Predictor', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('decision_tree.png', dpi=300)
print("✓ Saved: decision_tree.png")
print()

# ============================================================================
# STEP 6: RANDOM FOREST REGRESSOR
# ============================================================================

print("="*80)
print("STEP 6: Training Random Forest Model...")
print("="*80)
print()

rf_model = RandomForestRegressor(n_estimators=10, max_depth=2, random_state=42, min_samples_leaf=1)
rf_model.fit(X_train, y_train)

# Make predictions
y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)

# Evaluate
rf_train_r2 = r2_score(y_train, y_train_pred_rf)
rf_test_r2 = r2_score(y_test, y_test_pred_rf)
rf_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_rf))
rf_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))
rf_train_mae = mean_absolute_error(y_train, y_train_pred_rf)
rf_test_mae = mean_absolute_error(y_test, y_test_pred_rf)

print("RANDOM FOREST RESULTS:")
print("-"*80)
print(f"\nTraining Performance:")
print(f"  R² Score: {rf_train_r2:.4f}")
print(f"  RMSE: {rf_train_rmse:.2f} percentage points")
print(f"  MAE: {rf_train_mae:.2f} percentage points")
print(f"\nTest Performance:")
print(f"  R² Score: {rf_test_r2:.4f}")
print(f"  RMSE: {rf_test_rmse:.2f} percentage points")
print(f"  MAE: {rf_test_mae:.2f} percentage points")
print()

# Feature importance
rf_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Feature Importance (Random Forest):")
print(rf_importance.to_string(index=False))
print()

# ============================================================================
# STEP 7: MODEL COMPARISON
# ============================================================================

print("="*80)
print("STEP 7: Model Comparison...")
print("="*80)
print()

comparison_df = pd.DataFrame({
    'Model': ['Decision Tree', 'Decision Tree', 'Random Forest', 'Random Forest'],
    'Dataset': ['Train', 'Test', 'Train', 'Test'],
    'R² Score': [dt_train_r2, dt_test_r2, rf_train_r2, rf_test_r2],
    'RMSE': [dt_train_rmse, dt_test_rmse, rf_train_rmse, rf_test_rmse],
    'MAE': [dt_train_mae, dt_test_mae, rf_train_mae, rf_test_mae]
})

print(comparison_df.to_string(index=False))
print()

# Save comparison
comparison_df.to_csv('model_comparison.csv', index=False)
print("✓ Saved: model_comparison.csv")
print()

# ============================================================================
# STEP 8: PREDICTION ON TEST DATA
# ============================================================================

print("="*80)
print("STEP 8: Predictions on test data...")
print("="*80)
print()

predictions_df = pd.DataFrame({
    'Actual': y_test.values,
    'DT_Prediction': y_test_pred_dt,
    'RF_Prediction': y_test_pred_rf,
    'DT_Error': y_test.values - y_test_pred_dt,
    'RF_Error': y_test.values - y_test_pred_rf
})

print("Test Set Predictions:")
print(predictions_df.to_string(index=False))
print()

predictions_df.to_csv('predictions.csv', index=False)
print("✓ Saved: predictions.csv")
print()

# ============================================================================
# STEP 9: VISUALIZATIONS
# ============================================================================

print("STEP 9: Creating visualizations...")
print("-"*80)

# Feature importance comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Decision Tree importance
axes[0].barh(dt_importance['Feature'], dt_importance['Importance'], color='steelblue')
axes[0].set_xlabel('Importance Score', fontsize=11)
axes[0].set_title('Feature Importance - Decision Tree', fontsize=12, fontweight='bold')
axes[0].invert_yaxis()

# Random Forest importance
axes[1].barh(rf_importance['Feature'], rf_importance['Importance'], color='coral')
axes[1].set_xlabel('Importance Score', fontsize=11)
axes[1].set_title('Feature Importance - Random Forest', fontsize=12, fontweight='bold')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
print("✓ Saved: feature_importance.png")

# Actual vs Predicted scatter plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Decision Tree
axes[0].scatter(y_test, y_test_pred_dt, s=100, alpha=0.6, color='steelblue')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
            'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Adoption (%)', fontsize=11)
axes[0].set_ylabel('Predicted Adoption (%)', fontsize=11)
axes[0].set_title(f'Decision Tree: Actual vs Predicted (R²={dt_test_r2:.3f})', 
                 fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Random Forest
axes[1].scatter(y_test, y_test_pred_rf, s=100, alpha=0.6, color='coral')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
            'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Adoption (%)', fontsize=11)
axes[1].set_ylabel('Predicted Adoption (%)', fontsize=11)
axes[1].set_title(f'Random Forest: Actual vs Predicted (R²={rf_test_r2:.3f})', 
                 fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300)
print("✓ Saved: actual_vs_predicted.png")

print()

# ============================================================================
# STEP 10: INTERPRETATION GUIDE
# ============================================================================

print("="*80)
print("STEP 10: How to interpret results...")
print("="*80)
print()

print("WHAT THE METRICS MEAN:")
print("-"*80)
print()
print("R² Score (Coefficient of Determination):")
print("  - Range: 0 to 1 (higher is better)")
print("  - 1.0 = Perfect prediction")
print("  - 0.5 = Model explains 50% of variation")
print("  - 0.0 = Model explains 0% of variation")
print(f"  - Your best model: R² = {max(dt_test_r2, rf_test_r2):.4f}")
print()

print("RMSE (Root Mean Square Error):")
print("  - Measures typical prediction error in same units as target")
print("  - Lower is better")
print(f"  - Your best model: RMSE = {min(dt_test_rmse, rf_test_rmse):.2f} percentage points")
print(f"  - Interpretation: On average, predictions off by ±{min(dt_test_rmse, rf_test_rmse):.1f}%")
print()

print("MAE (Mean Absolute Error):")
print("  - Average magnitude of errors")
print("  - Lower is better")
print(f"  - Your best model: MAE = {min(dt_test_mae, rf_test_mae):.2f} percentage points")
print()

print("FEATURE IMPORTANCE:")
print("-"*80)
print(f"\nTop feature (Decision Tree): {dt_importance.iloc[0]['Feature']}")
print(f"  → This variable has strongest effect on stablecoin adoption")
print()
print(f"Top feature (Random Forest): {rf_importance.iloc[0]['Feature']}")
print(f"  → This variable has strongest effect on stablecoin adoption")
print()

print("="*80)
print("FOR YOUR RESEARCH PAPER:")
print("="*80)
print()
print("Write something like:")
print()
print(f'"We trained two machine learning models (Decision Tree and Random Forest)')
print(f'on {len(X_train)} observations to predict stablecoin adoption.')
print(f'The {["Decision Tree", "Random Forest"][rf_test_r2 > dt_test_r2]} achieved')
print(f'R² = {max(dt_test_r2, rf_test_r2):.3f} on test data, explaining')
print(f'{max(dt_test_r2, rf_test_r2)*100:.1f}% of adoption variation.')
print(f'Feature importance analysis shows {max(dt_importance.iloc[0]["Feature"], rf_importance.iloc[0]["Feature"])}')
print(f'as the strongest predictor, suggesting [interpretation]."')
print()

print("="*80)
print("PREDICTIVE MODELING COMPLETE")
print("="*80)
print()
print("Output files:")
print("  ✓ decision_tree.png (tree visualization)")
print("  ✓ feature_importance.png (which variables matter most)")
print("  ✓ actual_vs_predicted.png (model accuracy visualization)")
print("  ✓ model_comparison.csv (performance metrics)")
print("  ✓ predictions.csv (test set predictions)")
print()
