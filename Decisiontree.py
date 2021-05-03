
#Here we call the essential libraries to help with our ML Algorithm
import matplotlib.pyplot as plt                             #necessary to help with our data visualization
import pandas as pd                                         #help with working on multi-dimensional arrays
from sklearn.tree import DecisionTreeClassifier             #the main libaries that will help with the learning/training
from sklearn import tree                                    #using this to build a decision tree based on the code

data_set = pd.read_csv("test_data/train.csv")               #read the data set by giving the function a path to where it is

clf =DecisionTreeClassifier()                               #this is our classifier that'll take in data and perform the

#training dataset
x_train = data_set.iloc[0:21000,1:].values                  #will use the first 21k rows, and will not use the first column because it includes the answer
train_label = data_set.iloc[0:21000,0].values               #train label is what our x_train will be comparing itself to
clf.fit(x_train,train_label)                                #now we call the DTC function and fit the x_train and train_label to it

#testing data
x_test = data_set.iloc[21000:,1:]                           #Now we will use the rest of the rows of data to test our training model
actual_label = data_set.iloc[21000:,0].values               #give it some information to check itself with

Pred=clf.predict(x_test)                                    #Now we want to start the prediction by using x_test

count=0                                                     #initializing a counter in order to be able to calculate accuracy
for i in range(0,21000):                                    #we start the for-loop and tell it how many times to run
    count +=1 if Pred[i] == actual_label[i] else 0          # will add 1 to the count if it guesses right,wont add if wrong

    print("Accuracy= ",(count/21000)*100)                   #print statement that'll print after each iteration or about 21000 times

fig = plt.figure(figsize=(10,10))                           #plt.fig will call on the matplotlib to construct a decision tree figure of the code above
F = tree.plot_tree(clf,filled=True)                         #clf is the classifier and the filled statement just assigns different colors to different nodes
fig.savefig("decision_tree.png")                            #save the figure as a png named decision_tree that is included in the report

text_representation = tree.export_text(clf)                 #just some extra way of visualize what each of the nodes say though it does get really messy and hard to follow
print(text_representation)