import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

# Extract the zip file
with zipfile.ZipFile('titanic.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('tested.csv')

# Clean and preprocess the data
# Drop columns that are not needed
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Fill missing values in the 'Age' column with the median age
median_age = df['Age'].median()
df['Age'].fillna(median_age, inplace=True)

#replace missing values in embarked
mode_embarked = df['Embarked'].mode()[0]
df['Embarked'].fillna(mode_embarked, inplace=True)

# Convert the 'Sex' column to binary values
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Family size new column
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Analyzzzee
# Calculate the mean age for male and female passengers
mean_age_male = df[df['Sex'] == 0]['Age'].mean()
mean_age_female = df[df['Sex'] == 1]['Age'].mean()
print(f"Mean age for male passengers: {mean_age_male:.2f}")
print(f"Mean age for female passengers: {mean_age_female:.2f}")
print(df.head())
sns.histplot(df['Age'], kde=False)
plt.title('Passenger Age Distribution')
plt.show()

# Show the survival rate by passenger class
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Passenger Class')
plt.show()

# Show survival rate by family size
sns.barplot(x='FamilySize', y='Survived', data=df)
plt.title('Survival Rate by Family Size')
plt.show()

# Create age groups
bins = [0, 12, 18, 30, 50, 80]
labels = ['Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

# Show the survival rate by age group
sns.barplot(x='AgeGroup', y='Survived', data=df)
plt.title('Survival Rate by Age Group')
plt.show()

# run this code to save new csv file once satisfied wth cleansing -- df.to_csv('modified.csv', index=False)


