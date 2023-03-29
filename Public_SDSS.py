# This Python Code applying DSS using Enhanced Decision tree classifier that work more accurate than regular DSS
# Also this code can apply in Spatial Data. let's try and let me know the feedback  

import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

df = pandas.read_csv("data.csv")

# If your decisions already include String then you need to change it to numeric as following 
d = {'No action needed': 0, 'Take Decision One': 1, 'Take Decision Two':2}
df['decision'] = df['decision'].map(d)


# write here your real features instead of these number Feature
features = ['Feature_one', 'feature two', 'feature three']

X = df[features]
y = df['decision']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Train
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train, y_train)


# Validate the model
from sklearn.metrics import accuracy_score
y_pred = dtree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100))

def recommend_control_measure(features):
    # Use the trained model to predict the number of cases
    predicted_cases = dtree.predict(features)
    
    # Determine the recommended control measure based on the predicted number of cases
    if predicted_cases < 100:
        return "No action needed"
    elif predicted_cases < 1000:
        return "Take Decision One"
    else:
        return "Take Decision Two"

# Generate a recommendation for a community with the following input features
community_features = [50000, 5000, 12000]
recommendation = recommend_control_measure([community_features])

# Results maybe changes every time you made run depending on your input in the community_features 
print(recommendation)







