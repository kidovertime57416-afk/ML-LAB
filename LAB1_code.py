import pandas as pd

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
