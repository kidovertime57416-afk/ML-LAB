from sklearn.preprocessing import LabelEncoder
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#Dataset Exploration
#1.Load Dataset
df = pd.read_csv(r"C:\Users\Kidovertime\Desktop\ML\LAB01\vgsales.csv")

#2.Display Shape
print("\n------------------------------")
print(df.shape)

#3.Display Data Types
print("\n------------------------------")
print(df.dtypes)

#4.Display Summary Statistics
print("\n------------------------------")
print(df.describe(include='all'))

#5.Display Missing Values
print("\n------------------------------")
print(df.isnull().sum())

#6.Display Duplicate Records
print("\n------------------------------")
print(df.duplicated().sum())

#7.Display Class Distribution
print("\n------------------------------")
print(df['Genre'].value_counts())

#Data Visualization
# 1. Histogram
plt.figure(figsize=(8, 5))
sns.histplot(df['Global_Sales'], bins=50, kde=True)
plt.title('Histogram of Global Sales')
plt.xlabel('Global Sales')
plt.ylabel('Frequency')
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

#Data Cleaning
# 1. Duplicate Removal
df_cleaned = df.drop_duplicates().copy()

print("--- ก่อนทำ Data Cleaning ---")
print(f"จำนวนข้อมูลทั้งหมด: {df.shape[0]} แถว")
print(f"ค่าว่างในคอลัมน์ Year: {df['Year'].isnull().sum()} ค่า")
print(f"ค่าว่างในคอลัมน์ Publisher: {df['Publisher'].isnull().sum()} ค่า")
print(f"จำนวนแถวที่ซ้ำซ้อน: {df.duplicated().sum()} แถว")
print("-" * 30)

# 2. Data Type Conversion & Incorrect Data Correction
df_cleaned['Year'] = pd.to_numeric(df_cleaned['Year'], errors='coerce')

# 3. Missing Value Handling & Compare
mean_year = df_cleaned['Year'].mean()
median_year = df_cleaned['Year'].median()

print("\n[เปรียบเทียบสถิติสำหรับคอลัมน์ Year]")
print(f"ค่าเฉลี่ย (Mean Year): {mean_year:.2f}")
print(f"ค่ามัธยฐาน (Median Year): {median_year:.1f}")

df_cleaned['Year'] = df_cleaned['Year'].fillna(median_year)

df_cleaned['Year'] = df_cleaned['Year'].astype(int)

mode_publisher = df_cleaned['Publisher'].mode()[0]
df_cleaned['Publisher'] = df_cleaned['Publisher'].fillna(mode_publisher)

print("\n--- หลังทำ Data Cleaning ---")
print(f"จำนวนข้อมูลคงเหลือ: {df_cleaned.shape[0]} แถว")
print(f"ค่าว่างในคอลัมน์ Year หลังแก้: {df_cleaned['Year'].isnull().sum()} ค่า")
print(f"ค่าว่างในคอลัมน์ Publisher หลังแก้: {df_cleaned['Publisher'].isnull().sum()} ค่า")

#Feature Engineering
#1.Label Encoding
le = LabelEncoder()
df_cleaned['Genre_LabelEncoded'] = le.fit_transform(df_cleaned['Genre'])

print("--- ผลลัพธ์การทำ Label Encoding ---")
print(df_cleaned[['Genre', 'Genre_LabelEncoded']].drop_duplicates().head(10))
print("-" * 30)

#2.One-Hot Encoding
df_onehot = pd.get_dummies(df_cleaned, columns=['Platform'], prefix='Platform', drop_first=True)

print("\n--- ผลลัพธ์การทำ One-Hot Encoding ---")
platform_cols = [col for col in df_onehot.columns if col.startswith('Platform_')]
print(df_onehot[['Name'] + platform_cols[:5]].head())
print("-" * 30)