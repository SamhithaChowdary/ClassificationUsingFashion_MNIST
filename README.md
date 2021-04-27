## ClassificationUsingFashion_MNIST
I have done this kaggle competition with Sravani Subraveti

In this, we are given the Fashion MNIST Dataset published by The Zalando’s article consists of (60000,784) training data and (10000,784) testing data. Each sample is 28x28 gray-scale image associated with a label from 5 classes. The problem statement here is to classify the dataset and predict the labels of test dataset based on the train data.

In the first part, we used Feature extraction methods like PCA, LDA methods to reduce the dimensions of the large dataset which has 784 columns in both train and test dataset. After applying the dimentionality reduction techniques we applied classical classification techniques to the reduced data set obtained. We used classical methods like Support Vector Machine (SVM), Random Forest(RF), Gradient Boosting Classifier (GBC).

<img width="370" alt="kagglescore" src="https://user-images.githubusercontent.com/55220359/116316248-a5d7ca80-a77f-11eb-8762-1b58b53de883.png">
Initially, we when were implementing the PCA with SVM model, we tried to apply the algorithm to whole data, which was taking lot of time to execute the model. So, we have decided to implement the gridsearchCV to SVM model so that we can apply the best parameters to the data.
After, applying the gridsearchCV to the whole data, we still could not find any improvement in the time it was taking. After, giving it a thought we have decided to split the data and to train the model initially to get an idea about the best Hyperparameters. We have applied this technique of GridSearchCV and RandomSearchCV to SVM and Random forest. For Gradient Boosting Classifier, we have directly applied the best parameters based on Zalando’s research link. Eventually, we got an idea about how to apply the hyper-parameters to the models, and started our trial and error method to get highest accuracy for the model to predict image classification for Fashion-MNIST dataset.

In the second part, We applied the Basic CNN model with single layer, two layers and also with three layers on to the dataset. We have also applied Resnet with two different epochs.

<img width="350" alt="Screen Shot 2021-04-27 at 5 52 45 PM" src="https://user-images.githubusercontent.com/55220359/116317706-b38e4f80-a781-11eb-9434-169637d62b6e.png">

We built our model using the Keras framework. The reason for using Keras framework is that it is a high-level neural networks API and capable of running on top of Tensor flow, CNTK or Theano.

Comparison of different Algorithms and parameters we have tried:
• After implementing the models that mentioned above, we got the best accuracy for CNN with one convolutional layer having the parameters as epochs= 50, batch size= 128 and learning rate as default value(0.001). The accuracy of the model obtained is
88.08%. The run time for training and testing data is 68.54 minutes.
• The next best accuracy was obtained for CNN with two convolutional layer having the parameters as epochs=100, batch size=32 and with default learning rate. The model accuracy is 81.2%.
• For CNN with three convolutional layers with epochs=100, batch size= 64 and lr=0.0001, the accuracy is 80.48%.
• Finally, with the implementation of Resnet, we got an accuracy of 86.02% with parameters as epochs = 25, batch size = 100. The run time for training and testing data is 10.92 minutes.

Plots of Training epoch Vs loss and Training epoch Vs Accuracy for CNN based architecture:

<img width="340" alt="cnn_plots" src="https://user-images.githubusercontent.com/55220359/116319440-87280280-a784-11eb-9084-a2ce122f875b.png">

Plots of Training epoch Vs loss and Training epoch Vs Accuracy for ResNet(Residual Neural
Networks):

<img width="340" alt="resnet" src="https://user-images.githubusercontent.com/55220359/116319449-8becb680-a784-11eb-9e62-20a66723bf15.png">

ResNet architecture is based on CNN architecture, but it is more complex than basic CNN model. With one convolution layer of CNN, we have observed more accuracy. With less number of Epochs for CNN and ResNet, we are observing more accuracy. The batch size of the model is also selected moderately, so that the model can train on the data properly, to get better accuracy on the test dataset. Usually, lower value of Learning Rate reduces the overfitting of the model. However, in this case we have taken the default value i.e, 0.001 for the Adam Optimizer. The run time depends on epochs, we have observed more time for CNN because, the epochs are more. Although, for ResNet the epochs are less, hence the run time is less. It would give more accuracy by changing the hyperparameters of the model. Nevertheless, we have implemented the CNN model with varied parameter values compared to ResNet. Therefore, For the image classification of Fashion MNIST dataset we have observed more accuracy for CNN model with one convolutional layer which is 88.08%.

References

[1] https://www.machinecurve.com/index.php/2019/09/17/how-to-create-a-cnn-classifierwith-keras/

[2] https://pravarmahajan.github.io/fashion/

[3] https://github.com/cmasch/zalando-fashion-mnist

[4] https://github.com/zalandoresearch/fashion-mnist

[5] http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/

[6] https://github.com/nuclearczy/SVM_and_CNN_on_Fashion_MNIST_Dataset

[7] https://github.com/shoji9x9/Fashion-MNIST-By-ResNet/blob/master/FashionMNIST-by-ResNet-50.ipynb

[8] https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras446d7ff84d33

[9] https://www.kaggle.com/girishgupta/fashion-mnist-using-resnet

[10] https://towardsdatascience.com/introduction-to-resnets-c0a830a288a4

[11] https://stackoverflow.com/questions/54589669/confusion-matrix-error-classificationmetrics-cant-handle-a-mix-of-multilabel

[12] https://www.kaggle.com/akumaldo/resnet-from-scratch-keras

[13] https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithmf10ba6e38234

[14] https://towardsdatascience.com/the-4-convolutional-neural-network-models-that-canclassify-your-fashion-images-9fe7f3e5399d

[15] https://keras.io/

[16] https://bmcproc.biomedcentral.com/articles/10.1186/1753-6561-5-S3-S11

[17] https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras446d7ff84d33

[18] https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-ondeep-learning-neural-networks/

[19] https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9

[20] https://www.kaggle.com/faressayah/fashion-classification-mnist-cnn-tutorial
