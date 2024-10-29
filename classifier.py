import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df = pd.read_csv('student_grades_UniOfJordan.csv')
encoder = OrdinalEncoder()
X = df[['Subject1', 'Subject2', 'Subject3', 'Subject4']]
X_encoded = encoder.fit_transform(X)
y = df['Result']
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
clf = CategoricalNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"{accuracy:.2%}")
new_student = pd.DataFrame({
    'Subject1': ['A'],
    'Subject2': ['B'],
    'Subject3': ['C'],
    'Subject4': ['F']
})
new_student_encoded = encoder.transform(new_student)
prediction = clf.predict(new_student_encoded)
print(f"{dict(new_student.iloc[0])}")
print({prediction[0]})
