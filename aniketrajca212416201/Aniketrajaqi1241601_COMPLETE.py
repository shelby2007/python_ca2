# Project: Air Quality Index (AQI) Analysis of India




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats




df = pd.read_excel("AQI.xlsx")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes)



# STEP 2 - Check Missing Values


print("\nMissing values in each column:")
print(df.isnull().sum())


df = df.dropna(subset=['pollutant_avg', 'pollutant_min', 'pollutant_max'])
print("\nRows after removing missing values:", len(df))



#Summary Statistics


print("\nSummary Statistics:")
print(df[['pollutant_min', 'pollutant_avg', 'pollutant_max']].describe())

print("\nMean:", round(df['pollutant_avg'].mean(), 2))
print("Median:", round(df['pollutant_avg'].median(), 2))
print("Std Dev:", round(df['pollutant_avg'].std(), 2))
print("Skewness:", round(df['pollutant_avg'].skew(), 4))



# OBJECTIVE 1 - Bar Chart: Average Pollutant Levels


avg_by_pollutant = df.groupby('pollutant_id')['pollutant_avg'].mean().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
plt.bar(avg_by_pollutant.index, avg_by_pollutant.values, color='steelblue', edgecolor='black')
plt.title('Average Level of Each Pollutant')
plt.xlabel('Pollutant Type')
plt.ylabel('Average Value')
plt.tight_layout()
plt.savefig('graph1_avg_pollutant.png', dpi=150)
plt.close()
print("Saved: graph1_avg_pollutant.png")



# OBJECTIVE 2 - Histogram: Distribution of PM2.5


pm25 = df[df['pollutant_id'] == 'PM2.5']['pollutant_avg']

plt.figure(figsize=(7, 5))
plt.hist(pm25, bins=20, color='coral', edgecolor='black')
plt.axvline(pm25.mean(), color='blue', linestyle='--', label=f'Mean = {pm25.mean():.1f}')
plt.axvline(pm25.median(), color='green', linestyle='--', label=f'Median = {pm25.median():.1f}')
plt.title('Distribution of PM2.5 Pollutant')
plt.xlabel('PM2.5 Average Value')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('graph2_pm25_histogram.png', dpi=150)
plt.close()
print("Saved: graph2_pm25_histogram.png")



# OBJECTIVE 3 - Box Plot: Outlier Detection


plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x='pollutant_id', y='pollutant_avg', palette='Set2')
plt.title('Pollutant Distribution and Outliers (Box Plot)')
plt.xlabel('Pollutant Type')
plt.ylabel('Average Value')
plt.tight_layout()
plt.savefig('graph3_boxplot_outliers.png', dpi=150)
plt.close()
print("Saved: graph3_boxplot_outliers.png")

# IQR Method to count outliers
Q1 = df['pollutant_avg'].quantile(0.25)
Q3 = df['pollutant_avg'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['pollutant_avg'] < Q1 - 1.5 * IQR) | (df['pollutant_avg'] > Q3 + 1.5 * IQR)]
print(f"\nOutliers found using IQR method: {len(outliers)}")



# OBJECTIVE 4 - Correlation Heatmap


# Select only numeric columns for correlation
numeric_cols = df[['pollutant_min', 'pollutant_avg', 'pollutant_max', 'latitude', 'longitude']]
corr_matrix = numeric_cols.corr()

print("\nCorrelation Matrix:")
print(corr_matrix.round(2))

plt.figure(figsize=(7, 5))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='Blues', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('graph4_correlation_heatmap.png', dpi=150)
plt.close()
print("Saved: graph4_correlation_heatmap.png")



# OBJECTIVE 5 - Top 10 Most Polluted States


# Group by state and calculate mean pollutant_avg
top_states = df.groupby('state')['pollutant_avg'].mean().sort_values(ascending=False).head(10)

print("\nTop 10 Most Polluted States:")
print(top_states)

plt.figure(figsize=(9, 6))
top_states.sort_values().plot(kind='barh', color='tomato', edgecolor='black')
plt.title('Top 10 Most Polluted States')
plt.xlabel('Average Pollutant Level')
plt.ylabel('State')
plt.tight_layout()
plt.savefig('graph5_top_polluted_states.png', dpi=150)
plt.close()
print("Saved: graph5_top_polluted_states.png")


# ============================================================
# OBJECTIVE 6 - Hypothesis Testing (Independent t-test)


# Filter PM2.5 and PM10 data
pm25_data = df[df['pollutant_id'] == 'PM2.5']['pollutant_avg']
pm10_data = df[df['pollutant_id'] == 'PM10']['pollutant_avg']

# Perform independent samples t-test
t_stat, p_value = stats.ttest_ind(pm25_data, pm10_data)

print("\n--- Hypothesis Testing (t-test) ---")
print(f"PM2.5 Mean: {pm25_data.mean():.2f}")
print(f"PM10 Mean:  {pm10_data.mean():.2f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.6f}")

alpha = 0.05
if p_value < alpha:
    print("Decision: Reject H0 - Significant difference exists between PM2.5 and PM10")
else:
    print("Decision: Fail to Reject H0 - No significant difference")



# OBJECTIVE 7 - Linear Regression (Machine Learning)


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Step 1 - Select Features and Target
X = df[['pollutant_min', 'pollutant_max']]   # Input features
y = df['pollutant_avg']                       # Target variable

# Step 2 - Split into Train (80%) and Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining Data Size: {len(X_train)} rows")
print(f"Testing Data Size:  {len(X_test)} rows")

# Step 3 - Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4 - Predict on test data
y_pred = model.predict(X_test)

# Step 5 - Evaluate the model
r2  = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\nR2 Score: {round(r2, 4)}")
print(f"Mean Squared Error (MSE): {round(mse, 2)}")

# Step 6 - Plot Actual vs Predicted
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, color='violet', alpha=0.5, edgecolors='purple', linewidths=0.3)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linestyle='--', linewidth=1.5)
plt.title('Linear Regression: Actual vs Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.tight_layout()
plt.savefig('graph6_linear_regression.png', dpi=150)
plt.close()
print("Saved: graph6_linear_regression.png")

print("All objectives completed successfully!")
print("Graphs saved: graph1 to graph6")

