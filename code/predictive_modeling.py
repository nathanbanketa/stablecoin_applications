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


df = pd.read_csv("merged_dataset_for_regression.csv")

print(f"✓ Loaded data: {df.shape[0]} observations")
print("\nColumns available:")
print(df.columns.tolist())
print()

# Drop rows with missing dependent variable (adoption rate)
df_clean = df.dropna(subset=['Stablecoin_Share_Pct'])

print(f"After removing missing DV: {df_clean.shape[0]} observations")
print()



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


print("\nFeature Summary Statistics:")
print(X_train.describe().to_string())
print()

print("Target (Stablecoin Share %) Statistics:")
print(f"  Mean: {y_train.mean():.2f}%")
print(f"  Std Dev: {y_train.std():.2f}%")
print(f"  Min: {y_train.min():.2f}%")
print(f"  Max: {y_train.max():.2f}%")
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