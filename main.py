import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import id3


data = pd.read_csv("lenses.data.txt", dtype=None,
                       names=["number", "age", "spectacle", "astigmatic", "tears", "class"],
                       delim_whitespace=True)
data = data.transform(lambda x: (x - 1))
target = data["class"]
data = data.drop("class", axis=1).drop("number", axis=1)
featureNames = ["age", "spectacle", "astigmatic", "tears"]

# split into training and test sets
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=.3, train_size=.7, shuffle=True)

classifier = id3.Id3Classifier()
model = classifier.fit(data_train, target_train, featureNames)
predictions = model.predict(data_test)
score = accuracy_score(target_test, predictions)
print("Custom ID3 tree with Lenses data")
print("Score: " + ("%.1f" % (score * 100)) + "%")