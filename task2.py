import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = 'titanic.csv'  
titanic_df = pd.read_csv(file_path)


titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].median())  # Fill Age with median
titanic_df = titanic_df.drop(columns=['Cabin'])  # Drop Cabin column due to many missing values
titanic_df['Embarked'] = titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0])  # Fill Embarked with mode


titanic_df = titanic_df.drop(columns=['PassengerId', 'Name', 'Ticket'])


titanic_df = pd.get_dummies(titanic_df, columns=['Sex', 'Embarked'], drop_first=True)


sns.set(style="whitegrid")


plt.figure(figsize=(6, 4))
sns.countplot(data=titanic_df, x='Survived')
plt.title('Survival Count')
plt.xlabel('Survived (1 = Yes, 0 = No)')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(10, 5))
sns.histplot(titanic_df['Age'], kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(8, 5))
sns.barplot(data=titanic_df, x='Pclass', y='Survived')
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class (1 = First, 2 = Second, 3 = Third)')
plt.ylabel('Survival Rate')
plt.show()


plt.figure(figsize=(6, 4))
sns.barplot(data=titanic_df, x='Sex_male', y='Survived')
plt.title('Survival Rate by Gender')
plt.xlabel('Gender (1 = Male, 0 = Female)')
plt.ylabel('Survival Rate')
plt.show()


plt.figure(figsize=(10, 5))
sns.kdeplot(titanic_df[titanic_df['Survived'] == 1]['Age'], fill=True, label='Survived')
sns.kdeplot(titanic_df[titanic_df['Survived'] == 0]['Age'], fill=True, label='Not Survived')
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend()
plt.show()



plt.figure(figsize=(10, 8))
sns.heatmap(titanic_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.show()
