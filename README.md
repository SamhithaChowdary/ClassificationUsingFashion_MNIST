## ClassificationUsingFashion_MNIST
I have done this kaggle competition with Sravani Subraveti

In this, we are given the Fashion MNIST Dataset published by The Zalando’s article consists of (60000,784) training data and (10000,784) testing data. Each sample is 28x28 gray-scale image associated with a label from 5 classes. The problem statement here is to classify the dataset and predict the labels of test dataset based on the train data.

In the first part, we used Feature extraction methods like PCA, LDA methods to reduce the dimensions of the large dataset which has 784 columns in both train and test dataset. After applying the dimentionality reduction techniques we applied classical classification techniques to the reduced data set obtained. We used classical methods like Support Vector Machine (SVM), Random Forest(RF), Gradient Boosting Classifier (GBC).

<img width="370" alt="kagglescore" src="https://user-images.githubusercontent.com/55220359/116316248-a5d7ca80-a77f-11eb-8762-1b58b53de883.png">
Initially, we when were implementing the PCA with SVM model, we tried to apply the algorithm to whole data, which was taking lot of time to execute the model. So, we have decided to implement the gridsearchCV to SVM model so that we can apply the best parameters to the data.
After, applying the gridsearchCV to the whole data, we still could not find any improvement in the time it was taking. After, giving it a thought we have decided to split the data and to train the model initially to get an idea about the best Hyperparameters. We have applied this technique of GridSearchCV and RandomSearchCV to SVM and Random forest. For Gradient Boosting Classifier, we have directly applied the best parameters based on Zalando’s research link. Eventually, we got an idea about how to apply the hyper-parameters to the models, and started our trial and error method to get highest accuracy for the model to predict image classification for Fashion-MNIST dataset.

In the second part, We applied the Basic CNN model with single layer, two layers and also with three layers on to the dataset. We have also applied Resnet with two different epochs.
