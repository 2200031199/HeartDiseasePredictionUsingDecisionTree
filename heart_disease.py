import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
train_df = pd.read_csv('heartdisease_train.csv')
test_df = pd.read_csv('heartdiseases_test.csv')
# Step 2: Preprocess the data# Separate features and target
X = train_df[['chest pain type', 'resting bp s', 'cholesterol', 'fasting blood sugar', 'resting ecg','max heart rate']]
y = train_df['target']
# Step 3: Train a machine learning model
X_train, X_test, y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model =DecisionTreeClassifier(criterion="entropy",random_state=100,max_depth=3,min_samples_leaf=5)
model.fit(X_train, y_train)
# Evaluate the model on validation sety_pred = model.predict(X_test)
text_representation=tree.export_text(model)
print(text_representation)
plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=['chest pain type', 'resting bp s', 'cholesterol', 'fasting blood sugar', 'resting ecg','max heart rate'], class_names=['target0', 'target1'])
plt.show()