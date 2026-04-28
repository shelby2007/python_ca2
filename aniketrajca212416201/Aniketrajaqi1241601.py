# Project: Air Quality Index (AQI) Analysis of India


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# STEP 1 - Load the dataset


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

# Drop rows where pollutant values are missing
df = df.dropna(subset=['pollutant_avg', 'pollutant_min', 'pollutant_max'])
print("\nRows after removing missing values:", len(df))



# STEP 3 - Summary Statistics


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

Q1 = df['pollutant_avg'].quantile(0.25)
Q3 = df['pollutant_avg'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['pollutant_avg'] < Q1 - 1.5 * IQR) | (df['pollutant_avg'] > Q3 + 1.5 * IQR)]
print(f"\nOutliers found using IQR method: {len(outliers)}")
