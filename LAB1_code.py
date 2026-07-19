import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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