# # Task 4: Prediction using Decision Tree Algorithum
# # Author: Gadade Kailas Rayappa
# Importing the required Libraries
import numpy as np
import pandas as pd 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# # Step 1: Import the dataset
iris_data=pd.read_csv("C:\\Users\\Kailas\\OneDrive\\Desktop\\Data II.csv")
iris_data.head()
# # Step 2: Exploratory Data Analysis
iris_data.shape
iris_data.info()
# unique value in each columns
for i in iris_data.columns:
    print(i, "\t\t",len(iris_data[i].unique()))
list_columns=iris_data.columns
list_columns
iris_data.tail()
iris_data.describe()
iris_data.isnull().sum()
iris_data.isnull().values.any()
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
x= iris_data.loc[:, features].values
print(x)
y=iris_data.Species
print(y)
# # Step 3: Data Visualization comparing various features
sns.pairplot(iris_data)
plt.figure(figsize=(12,8))
sns.heatmap(iris_data.describe(),annot=True,fmt='.2f',cmap='rainbow')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("visualize average,number0f,min,max,std,Queartile",fontsize=16)
# Relationship between the data
iris_data.corr()
plt.figure(figsize=(12,8))
sns.heatmap(iris_data.corr(),annot=True,fmt='.2f',cmap='rainbow')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("visualize average,number0f,min,max,std,Queartile",fontsize=16)
# # Step 4: Decision Tree Tree Model Training
# Model Training
x_train, x_test, y_train, y_test= train_test_split(x, y, random_state=0)
clf = DecisionTreeClassifier(max_depth = 2, random_state = 0)
clf.fit(x_train, y_train)
clf.predict(x_test[0:1])
from sklearn import preprocessing
iris_data['Specices']=preprocessing.LabelEncoder().fit_transform(iris_data['Species'])
iris_data.dtypes
X=iris_data.iloc[:,1:5].values
Y=iris_data.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,Y_train)
print("The Decision Tree Classification model trained")
classifier_tree=tree.DecisionTreeClassifier()
classifier_tree=classifier_tree.fit(X_train,Y_train)
# text graph representation
text_presentation=tree.export_text(classifier_tree)
print(text_presentation)
plt.figure(figsize=(18,16))
tree.plot_tree(classifier_tree,filled=True,impurity=True)
# # Step 5: Calculating the Model accuracy
y_pred=classifier.predict(X_test)
y_pred
print("Accuracy score:",np.mean(y_pred==y_test))
# Making the confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
score=np.mean(y_pred==y_test)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True,fmt='.0f',linewidths=0.5,square= True,cmap = 'ocean');
plt.ylabel('Actual label\n',fontsize = 14);
plt.xlabel('Predicted label\n', fontsize = 14);
plt.title('Accuracy Score: {}\n'.format(score),size =16);
plt.tick_params(labelsize= 15)
plt.show()
cm_accuracy=accuracy_score(y_test,y_pred)
print("Accuracy of model:",cm_accuracy)
# # Classification_report
from sklearn import metrics
print(metrics.classification_report(y_test, classifier.predict(X_test)))
# Conclusion
## The Classifier model can predict the Species of the flower 96% Accuracy score.
