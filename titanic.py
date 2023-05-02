import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import zipfile
import scipy.stats as stats

# Extract the zip file
with zipfile.ZipFile('titanic.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('tested.csv')

# CLEAN AND PREPROCESS THE DATA

# Drop columns that are not needed

df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# fill missing values in the 'Age' and 'Fare' column with the median age
median_age = df['Age'].median()
df['Age'].fillna(median_age, inplace=True)

median_fare = df['Fare'].median()
df['Fare'].fillna(median_fare, inplace=True)

#replace missing values in embarked
mode_embarked = df['Embarked'].mode()[0]
df['Embarked'].fillna(mode_embarked, inplace=True)

# convert the 'Sex' column to binary values
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# family size new column
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# new fare column with fare groups
df['FareNew'] = df['Fare'].round().astype(int)
bins = [0, 10, 20, 30, 40, 50, 100, 1000]
labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-100', '>100']
df['FareGroup'] = pd.cut(df['FareNew'], bins=bins, labels=labels)


# ANALYZE THE DATA

# print the head with the changes in the dataset
print(df.head())

# number of passengers in each passenger class

print("Passenger class counts:\n", df['Pclass'].value_counts())

# number of passengers by sex and plot the data

passengers_by_sex = df.groupby('Sex')['PassengerId'].count()

print("Passengers by sex:\n",  df.groupby('Sex')['PassengerId'].count())

sns.barplot(x=passengers_by_sex.index, y=passengers_by_sex.values)
plt.title('Number of Passengers by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# calculate the mean age for male and female passengers
mean_age_male = df[df['Sex'] == 0]['Age'].mean()
mean_age_female = df[df['Sex'] == 1]['Age'].mean()
print(f"Mean age for male passengers: {mean_age_male:.2f}")
print(f"Mean age for female passengers: {mean_age_female:.2f}")

# plot age distribution
sns.histplot(df['Age'], kde=False)
plt.title('Passenger Age Distribution')
plt.show()

# show the survival rate by passenger class
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Passenger Class')
plt.show()



# show survival rate by family size
sns.barplot(x='FamilySize', y='Survived', data=df)
plt.title('Survival Rate by Family Size')
plt.show()

# create age groups and survival rate per age groups
bins = [0, 12, 18, 30, 50, 80]
labels = ['Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

sns.barplot(x='AgeGroup', y='Survived', data=df)
plt.title('Survival Rate by Age Group')
plt.show()


# survival rate by port they mounted from
sns.barplot(x='Embarked', y='Survived', data=df)
plt.title('Survival Rate by Port of Embarkation')
plt.show()

# correlation between port of embarkation and survival rate + p-value
embarked_values = df['Embarked'].unique()
p_values = []
for value in embarked_values:
    data = df[df['Embarked'] == value]['Survived']
    p_value = stats.ttest_1samp(data, df['Survived'].mean())[1]
    p_values.append(p_value)
    
for i in range(len(embarked_values)):
    print(f"p-value for {embarked_values[i]}: {p_values[i]:.10f}")

# mean survival for each class
survival_by_class = df.pivot_table(values='Survived', index='Pclass')
print(survival_by_class)

# pearson correlation coefficient between SibSp and Parch
corr = np.corrcoef(df['SibSp'], df['Parch'])[0, 1]
print(f"Pearson correlation coefficient between SibSp and Parch: {corr:.2f}")

# correlation between gender and survival rate with chi-square test and print p-value
contingency_table = pd.crosstab(df['Sex'], df['Survived'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print("p-value for relationship between gender and survival rate:", p)

# run code below to save new csv file if need
# df.to_csv('modified.csv', index=False)


