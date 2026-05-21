

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os, warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "./a2_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================
# PHASE 1: LOAD UNIFIED DATASET
# =====================================================================
print("=" * 70)
print("PHASE 1: LOADING UNIFIED DATASET (2019-2025)")
print("=" * 70)

df_raw = pd.read_csv("unified_tourism_data_2019_2025.csv")
print(f"Raw dataset: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")
print(f"Years: {sorted(df_raw['year'].unique())}")
print(f"Data levels: {df_raw['data_level'].value_counts().to_dict()}")


# =====================================================================
# PHASE 2: PREPROCESSING (8 Steps)
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 2: DATA PREPROCESSING")
print("=" * 70)

# ----- Step 1: Missing Values Analysis -----
print("\n--- Step 1: Missing Values Analysis (Raw Data) ---")
print(f"Total rows: {len(df_raw)}")
for col in df_raw.columns:
    missing = df_raw[col].isna().sum()
    if missing > 0:
        pct = missing / len(df_raw) * 100
        print(f"  {col:<40} {missing:>3} missing ({pct:.1f}%)")

# Chart: Missing values bar chart
fig, ax = plt.subplots(figsize=(12, 6))
missing_pct = (df_raw.isnull().sum() / len(df_raw) * 100).sort_values(ascending=True)
colors = ['#e74c3c' if x > 50 else '#f39c12' if x > 0 else '#2ecc71' for x in missing_pct]
missing_pct.plot(kind='barh', ax=ax, color=colors, edgecolor='black')
ax.set_xlabel('Missing Values (%)')
ax.set_title('Missing Values Analysis - Raw Unified Dataset (2019-2025)', fontweight='bold')
for i, (val, name) in enumerate(zip(missing_pct, missing_pct.index)):
    if val > 0:
        ax.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_missing_values_raw.png', dpi=150)
plt.close()

# Chart: Missing by year
print("\nBreakdown by year:")
fig, ax = plt.subplots(figsize=(12, 6))
year_data = []
for year in sorted(df_raw['year'].unique()):
    yr = df_raw[df_raw['year'] == year]
    occ_nan = yr['occ_hotel'].isna().sum()
    reg_nan = yr['region'].isna().sum()
    year_data.append({'Year': int(year), 'Rows': len(yr), 'occ_nan': occ_nan, 'reg_nan': reg_nan})
    print(f"  {year}: {len(yr)} rows | occ_hotel NaN: {occ_nan} | region NaN: {reg_nan}")

df_ym = pd.DataFrame(year_data)
x = range(len(df_ym))
width = 0.35
ax.bar([i - width/2 for i in x], df_ym['occ_nan'], width, label='Occupancy (NaN)', color='#e74c3c', edgecolor='black')
ax.bar([i + width/2 for i in x], df_ym['reg_nan'], width, label='Region (NaN)', color='#3498db', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(df_ym['Year'].astype(int))
ax.set_xlabel('Year'); ax.set_ylabel('Missing Values')
ax.set_title('Missing Values by Year: Occupancy vs Region', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_missing_by_year.png', dpi=150)
plt.close()


# ----- Step 2: Feature Selection (drop columns >90% missing) -----
print("\n--- Step 2: Feature Selection ---")
drop_cols = []
for col in df_raw.columns:
    pct = df_raw[col].isna().sum() / len(df_raw) * 100
    if pct > 90:
        drop_cols.append(col)
        print(f"  DROPPED: {col} ({pct:.1f}% missing)")

df = df_raw.drop(columns=drop_cols).copy()
print(f"Columns reduced: {df_raw.shape[1]} -> {df.shape[1]}")


# ----- Step 3: Handle Missing Values -----
print("\n--- Step 3: Handling Missing Values ---")

# 3a: Drop rows where target variable is NaN (2019-2020)
target_nan = df['occ_hotel'].isna().sum()
print(f"Rows with missing target (occ_hotel): {target_nan}")
print(f"  From years: {df[df['occ_hotel'].isna()]['year'].unique()}")
print(f"  Decision: DROP (cannot impute the target variable)")
df = df.dropna(subset=['occ_hotel'])
print(f"  Rows after drop: {len(df)}")

# 3b: Fill missing region with 'National'
region_nan = df['region'].isna().sum()
print(f"Rows with missing region: {region_nan}")
print(f"  From years: {df[df['region'].isna()]['year'].unique()}")
print(f"  Decision: FILL with 'National' label")
df['region'] = df['region'].fillna('National')

# 3c: Derive missing quarter from month
quarter_nan = df['quarter'].isna().sum()
print(f"Rows with missing quarter: {quarter_nan}")
print(f"  Decision: DERIVE from month_num")
def get_quarter(m):
    if pd.isna(m): return np.nan
    m = int(m)
    if m <= 3: return 'Q1'
    elif m <= 6: return 'Q2'
    elif m <= 9: return 'Q3'
    else: return 'Q4'

df['quarter'] = df.apply(lambda r: r['quarter'] if pd.notna(r['quarter']) else get_quarter(r['month_num']), axis=1)

print(f"\nAfter handling: {df.isnull().sum().sum()} total missing values")


# ----- Step 4: Duplicate Detection -----
print("\n--- Step 4: Duplicate Detection ---")
dupes = df.duplicated(subset=['year', 'month_num', 'region']).sum()
print(f"Duplicates found: {dupes}")


# ----- Step 5: Outlier Detection (IQR) -----
print("\n--- Step 5: Outlier Detection (IQR Method) ---")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, col in enumerate(['occ_hotel', 'adr_hotel', 'los_hotel']):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"  {col}: {len(outliers)} outliers (range: {lower:.3f} to {upper:.3f})")
    if len(outliers) > 0:
        print(f"    Regions: {list(outliers['region'].unique())}")
        print(f"    Decision: KEEP (real demand peaks)")
    
    bp = axes[i].boxplot(df[col].dropna(), vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor(['#3498db', '#2ecc71', '#9467bd'][i])
    bp['boxes'][0].set_alpha(0.7)
    axes[i].set_title(col.replace('_', ' ').title(), fontweight='bold')

plt.suptitle('Outlier Detection: Box Plots of Hotel Metrics', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_outlier_boxplots.png', dpi=150)
plt.close()


# ----- Step 6: Data Transformation (Log ADR) -----
print("\n--- Step 6: Data Transformation ---")
print(f"ADR skewness before: {df['adr_hotel'].skew():.3f}")
df['log_adr_hotel'] = np.log1p(df['adr_hotel'])
df['log_adr_apt'] = np.log1p(df['adr_apt'])
print(f"ADR skewness after:  {df['log_adr_hotel'].skew():.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.hist(df['adr_hotel'].dropna(), bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
ax1.set_title('Hotel ADR - Original', fontweight='bold')
ax1.set_xlabel('ADR (SAR)'); ax1.set_ylabel('Frequency')

ax2.hist(df['log_adr_hotel'].dropna(), bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
ax2.set_title('Hotel ADR - Log Transformed', fontweight='bold')
ax2.set_xlabel('Log(ADR)'); ax2.set_ylabel('Frequency')

plt.suptitle('Log Transformation: ADR Skewness Reduction', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_adr_log_transform.png', dpi=150)
plt.close()


# ----- Step 7: Feature Engineering -----
print("\n--- Step 7: Feature Engineering ---")

df['quarter_num'] = df['quarter'].str.replace('Q', '').astype(int)
df['sin_month'] = np.sin(2 * np.pi * df['month_num'] / 12)
df['cos_month'] = np.cos(2 * np.pi * df['month_num'] / 12)

# Hijri calendar features (approximate Gregorian mapping)
hajj_months = {2021: [7, 8], 2022: [7], 2023: [6, 7], 2024: [6], 2025: [6]}
ramadan_months = {2021: [4, 5], 2022: [4], 2023: [3, 4], 2024: [3, 4], 2025: [2, 3]}
riyadh_season = [10, 11, 12, 1, 2, 3]

df['is_hajj'] = df.apply(lambda r: 1 if int(r['month_num']) in hajj_months.get(int(r['year']), []) else 0, axis=1)
df['is_ramadan'] = df.apply(lambda r: 1 if int(r['month_num']) in ramadan_months.get(int(r['year']), []) else 0, axis=1)
df['is_riyadh_season'] = df['month_num'].apply(lambda m: 1 if int(m) in riyadh_season else 0)
df['is_summer'] = df['month_num'].apply(lambda m: 1 if int(m) in [7, 8] else 0)

# Region encoding
le = LabelEncoder()
df['region_encoded'] = le.fit_transform(df['region'])

print(f"New features: sin_month, cos_month, is_hajj, is_ramadan, is_riyadh_season, is_summer, region_encoded, quarter_num, log_adr_hotel, log_adr_apt")
print(f"Region mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
print(f"Dataset after engineering: {df.shape[0]} rows x {df.shape[1]} columns")

# Feature correlation
feature_cols = ['occ_hotel', 'occ_apt', 'adr_hotel', 'adr_apt', 'los_hotel', 'los_apt',
                'month_num', 'quarter_num', 'is_hajj', 'is_ramadan', 'is_riyadh_season',
                'is_summer', 'region_encoded', 'sin_month', 'cos_month']
fig, ax = plt.subplots(figsize=(13, 10))
corr = df[feature_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, square=True, linewidths=0.5)
ax.set_title('Feature Correlation Heatmap (After Preprocessing)', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_feature_correlation.png', dpi=150)
plt.close()


# ----- Step 8: Train-Test Split -----
print("\n--- Step 8: Train-Test Split ---")

# Save preprocessed data
df.to_csv('preprocessed_2019_2025.csv', index=False)

# Use regional data only for modeling
df_model = df[df['data_level'] == 'Regional'].copy()
print(f"Full preprocessed dataset: {len(df)} rows (2021-2025)")
print(f"Modeling subset (Regional): {len(df_model)} rows (2023-2025)")
print(f"Excluded from model: {len(df) - len(df_model)} national rows (2021-2022, no region)")

feature_names = ['log_adr_hotel', 'los_hotel', 'occ_apt', 'log_adr_apt',
                 'month_num', 'quarter_num', 'sin_month', 'cos_month',
                 'is_hajj', 'is_ramadan', 'is_riyadh_season', 'is_summer',
                 'region_encoded', 'year']

X = df_model[feature_names].values
y = df_model['occ_hotel'].values

train_mask = df_model['year'].isin([2023, 2024])
test_mask = df_model['year'] == 2025

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"Train: {X_train.shape[0]} samples (2023-2024)")
print(f"Test:  {X_test.shape[0]} samples (2025)")
print(f"Features: {len(feature_names)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Preprocessing pipeline summary chart
fig, ax = plt.subplots(figsize=(12, 6))
steps = ['1. Raw Data\n(428 rows, 15 cols)', '2. Drop Columns\n(>90% missing)',
         '3. Drop Target NaN\n(2019-2020)', '4. Fill Region NaN\n(\u2192 "National")',
         '5. Outlier Detection\n(IQR: keep all)', '6. Log Transform\n(ADR skew fix)',
         '7. Feature Engineering\n(+10 features)', '8. Train-Test Split\n(2023-24 / 2025)']
values = [428, 426, 426, 426, 426, 426, 426, 402]
cols_count = [15, 11, 11, 11, 11, 13, 21, 14]
step_colors = ['#e74c3c', '#e74c3c', '#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#2ecc71', '#9467bd']

ax.bar(range(len(steps)), values, color=step_colors, edgecolor='black', alpha=0.8)
ax.set_xticks(range(len(steps)))
ax.set_xticklabels(steps, fontsize=8, ha='center')
ax.set_ylabel('Number of Rows')
ax.set_title('Preprocessing Pipeline: Data Flow', fontweight='bold')
for i, (v, c) in enumerate(zip(values, cols_count)):
    ax.text(i, v + 5, f'{v} rows\n{c} cols', ha='center', fontsize=8, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/00_preprocessing_pipeline.png', dpi=150)
plt.close()


# =====================================================================
# PHASE 3: MODEL BUILDING
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 3: MODEL BUILDING & EVALUATION")
print("=" * 70)

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
}

results = {}
predictions = {}

print(f"\n{'Model':<25} {'RMSE':>8} {'MAE':>8} {'R\u00b2':>8} {'MAPE':>8}")
print("-" * 62)

for name, model in models.items():
    if name == 'Linear Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    results[name] = {'RMSE': rmse, 'MAE': mae, 'R\u00b2': r2, 'MAPE': mape}
    predictions[name] = y_pred
    print(f"{name:<25} {rmse:>8.4f} {mae:>8.4f} {r2:>8.4f} {mape:>7.1f}%")

# Cross-validation
print(f"\n--- 5-Fold Time Series Cross-Validation ---")
tscv = TimeSeriesSplit(n_splits=5)
cv_results = {}
for name, model in models.items():
    Xc = X_train_scaled if name == 'Linear Regression' else X_train
    scores = cross_val_score(model, Xc, y_train, cv=tscv, scoring='r2')
    cv_results[name] = scores
    print(f"  {name:<25} R\u00b2: {scores.mean():.4f} \u00b1 {scores.std():.4f}")

best_name = max(results, key=lambda x: results[x]['R\u00b2'])
print(f"\n*** Best Model: {best_name} (R\u00b2 = {results[best_name]['R\u00b2']:.4f}, MAPE = {results[best_name]['MAPE']:.1f}%) ***")


# =====================================================================
# PHASE 4: VISUALIZATIONS
# =====================================================================
print(f"\nGenerating charts...")
colors_m = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

# Model comparison
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for i, metric in enumerate(['RMSE', 'MAE', 'R\u00b2', 'MAPE']):
    vals = [results[m][metric] for m in models]
    bars = axes[i].bar(models.keys(), vals, color=colors_m, edgecolor='black', alpha=0.85)
    axes[i].set_title(metric, fontweight='bold', fontsize=13)
    axes[i].set_xticklabels(models.keys(), rotation=30, ha='right', fontsize=9)
    for bar, val in zip(bars, vals):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f'{val:.4f}' if metric != 'MAPE' else f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
plt.suptitle('Model Performance Comparison (Test Set 2025)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_model_comparison.png', dpi=150)
plt.close()

# Predicted vs Actual
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, (name, yp) in enumerate(predictions.items()):
    ax = axes[idx // 2][idx % 2]
    ax.scatter(y_test, yp, alpha=0.5, color=colors_m[idx], edgecolors='black', s=40)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect')
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    ax.set_title(f'{name} (R\u00b2={results[name]["R\u00b2"]:.3f})', fontweight='bold')
    ax.legend()
plt.suptitle('Predicted vs. Actual Occupancy Rate', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_predicted_vs_actual.png', dpi=150)
plt.close()

# Residuals
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, (name, yp) in enumerate(predictions.items()):
    ax = axes[idx // 2][idx % 2]
    ax.hist(y_test - yp, bins=25, color=colors_m[idx], alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', ls='--', lw=2)
    ax.set_xlabel('Residual'); ax.set_ylabel('Frequency')
    ax.set_title(f'{name} (MAE={results[name]["MAE"]:.4f})', fontweight='bold')
plt.suptitle('Residual Distribution', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_residual_distribution.png', dpi=150)
plt.close()

# CV box plot
fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot([cv_results[m] for m in models], labels=models.keys(), patch_artist=True)
for patch, c in zip(bp['boxes'], colors_m):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.set_ylabel('R\u00b2 Score')
ax.set_title('5-Fold Time Series Cross-Validation', fontweight='bold')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/09_cv_boxplot.png', dpi=150)
plt.close()

# Feature importance
best_model = models[best_name]
if hasattr(best_model, 'feature_importances_'):
    fi = pd.DataFrame({'Feature': feature_names, 'Importance': best_model.feature_importances_})
    fi = fi.sort_values('Importance', ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(fi['Feature'], fi['Importance'], color='#2ecc71', edgecolor='black', alpha=0.85)
    ax.set_xlabel('Importance')
    ax.set_title(f'Feature Importance ({best_name})', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/10_feature_importance.png', dpi=150)
    plt.close()
    
    print("\nFeature Importance:")
    for _, row in fi.sort_values('Importance', ascending=False).iterrows():
        print(f"  {row['Feature']:<25} {row['Importance']:.4f}")

# Actual vs Predicted for key regions
test_df = df_model[test_mask].copy()
test_df['predicted'] = predictions[best_name]

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
for i, region in enumerate(['Madinah', 'Riyadh', 'Makkah']):
    rdf = test_df[test_df['region'] == region].sort_values('month_num')
    axes[i].plot(rdf['month_num'], rdf['occ_hotel'] * 100, 'o-', color='#1f77b4', lw=2, ms=6, label='Actual')
    axes[i].plot(rdf['month_num'], rdf['predicted'] * 100, 's--', color='#e74c3c', lw=2, ms=6, label='Predicted')
    axes[i].set_ylabel('Occupancy (%)')
    axes[i].set_title(f'{region} - 2025', fontweight='bold')
    axes[i].legend()
axes[2].set_xticks(range(1, 10))
axes[2].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'])
plt.suptitle(f'Actual vs. Predicted ({best_name})', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/11_actual_vs_predicted_regions.png', dpi=150)
plt.close()

# Error by region
test_df['abs_error'] = np.abs(test_df['occ_hotel'] - test_df['predicted'])
region_errors = test_df.groupby('region')['abs_error'].mean().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 7))
region_errors.plot(kind='barh', ax=ax, color='#e74c3c', edgecolor='black', alpha=0.8)
ax.set_xlabel('Mean Absolute Error')
ax.set_title(f'Prediction Error by Region ({best_name})', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/12_error_by_region.png', dpi=150)
plt.close()


# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nPreprocessing:")
print(f"  Raw: 428 rows x 15 cols (2019-2025)")
print(f"  After preprocessing: 426 rows x 21 cols (2021-2025)")
print(f"  Dropped: 2 rows (2019-2020, target NaN)")
print(f"  Filled: 24 region NaN -> 'National'")
print(f"  Dropped columns: 4 (>99% missing)")
print(f"  Engineered: 10 new features")
print(f"\nModeling (Regional subset: 402 rows):")
print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"  Best: {best_name}")
print(f"    R\u00b2:   {results[best_name]['R\u00b2']:.4f}")
print(f"    RMSE: {results[best_name]['RMSE']:.4f}")
print(f"    MAPE: {results[best_name]['MAPE']:.1f}%")
print(f"\n13 charts saved to {OUTPUT_DIR}/")
print("DONE!")
