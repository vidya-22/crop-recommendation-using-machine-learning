# crop-recommendation-using-machine-learning

Agriculture is one of the important occupations practiced in India. More than 60% of the  land in the country is used for agriculture in order to suffice the needs of 1.3 billion  people. Thus, adapting to new agricultural technologies is very important. Crop growth  depends on various environmental factors like nitrogen, phosphorus, potassium in soil,  PH of soil, rainfall, temperature and humidity. The agriculture data consisting of various  crops with their respective environmental factors will help us understand the insights  and make an appropriate choice of crop using the above-mentioned factors. Our system  uses machine learning and suggests the best suitable crop for particular land based on soil  content and weather parameters. For this, we use the models- Logistic Regression,  Random Forest Regressor, Decision Tree Regressor, Gradient Boosting Regressor to obtain more accurate results while predicting the crop. This  project helps amateur farmers and gardeners to understand the interdependency of crops  on environmental factors and select a crop to grow based on their soil condition and  atmosphere in their region. 

 Dataset 

The dataset used for the system is the “Crop Recommendation Dataset” from the Kaggle repository. This dataset comprises of 8 attributes having information regarding 22 different crops with 100 records for each crop. In the above dataset, the input attributes are nitrogen(N), phosphorus(P), potassium(K) ratios in soil content, temperature in degree Celsius, humidity in percentage, rainfall in mm and the output attribute is the crop name. The data set consists of 7 attributes of numerical type and one attribute of nominal type. We convert the crop name into numerical value by using label encode method in the data pre-processing.   

 Following table gives the details regarding the various attributes for the dataset: 

No.     ATTRIBUTES       ATTRIBUTE TYPE 

1.      N                Numeric 

2.      P                Numeric 

3.      K                Numeric 

4.      Temperature      Numeric 

5.      Humidity         Numeric 

6.      Soil pH          Numeric 

7.      Rainfall         Numeric 

8.      Crop             String 


![Screenshot (1646)](https://user-images.githubusercontent.com/56502606/121810558-0616c300-cc7f-11eb-81e6-05a70f81fb9a.png)

![Screenshot (1643)](https://user-images.githubusercontent.com/56502606/121810548-f8f9d400-cc7e-11eb-8dad-55937109e690.png)


Logistic Regression 

Logistic regression is a regression model in this algorithm the discrete values are taken by target variables. The model is used for prediction and this function uses a sigmoid function. The accuracy of this model depends on two things those are precision and recall. Precision tells which portion was correct which has positive identifiers, and recall tells which portion was actually identified correctly with actual positives. 

Logistic regression is similar to that of linear regression but the difference between them is that logistic regression is used for classification problems and linear regression is used for regression problems. This regression technique can also do classification by determining the most effective variables. In this logistic regression we use a ‘S’ shaped curve, this curve is used to for prediction and is it used to predict values between 0 and 1. 


Random Forest Algorithm: 

Random forest is a bagging ensemble learning technique and is a supervised learning algorithm that is we provide the algorithm with input and their respective output while training. Bagging is a technique which fits multiple models on different subsets of a training dataset, then combines the predictions from all models. The accuracy of the model on test data depends on fine tuning of hyperparameters like number of decision trees, maximum depth of each tree, maximum features to consider at each split and random state. Random forest’s output is highly accurate because it considers the outputs of all the decision trees in the forest and calculates an average of all. It helps to solve a difficult and complex problem by controlling variance. Random forest uses bagging that also randomly selects subsets of features used in each data sample. Then further the decision is based on the majority of the votes of predictions. Thus, the final output is decided. 

Gradient Boosting Regressor 

Gradient boosting is a machine learning technique which is well known for its prediction speed and accuracy, when dealing with large and complex data. The name gradient boosting is given to the algorithm as target outcomes for each case are set based on the gradient of the error with respect to the prediction. It relies on the fact that the best possible next model, when merged with previous models, minimizes the overall prediction error. The central idea is to set the target outcomes for this next model in order to minimize the error. Each new model aims in the direction that decreases prediction error, in the room of possible predictions for each training case. 
We evaluate the predictions of a regression model using error metrics. We have calculated three types of errors.  


Decision Tree Regressor: 

Decision tree is a tree structured flowchart which contains nodes and each branch of the tree is used to represent the output from each attribute. The nodes are used for decision making and results of these decisions are given from the leaf nodes. With a specific data point, it is run completely through the entire tree by figuring out True/False questions till it reaches the leaf node. The final prediction is the average of the value of the dependent variable in that particular leaf node. By performing multiple iterations, the tree will predict an appropriate value for the data point. Decision tree is simple to understand but has a problem of overfitting to deal with. 


Types of errors are:

Mean absolute error (MAE)

MAE=abs(predicted-actual)/total test observations 

Mean square error (MSE)

MSE=(sigma(predicted-actual)/total test observations 

Root mean square error  (RMSE)

RMSE=sqrt(sigma(predicted-actual)/total test observations 

 The accuracy rates for the different models are shown in the following table for 75% training and 25% testing 
 
 Observations for Target: 2200 
Training Observations for Target: 1650 
Testing Observations for Target: 550

![Screenshot (1656)](https://user-images.githubusercontent.com/56502606/121810440-8be63e80-cc7e-11eb-8596-6174982cb25d.png)
![Screenshot (1647)](https://user-images.githubusercontent.com/56502606/121810450-999bc400-cc7e-11eb-8d38-855116458019.png)
![Screenshot (1648)](https://user-images.githubusercontent.com/56502606/121810472-a6b8b300-cc7e-11eb-89e2-9c1b1b37488e.png)
![Screenshot (1649)](https://user-images.githubusercontent.com/56502606/121810481-b0dab180-cc7e-11eb-9d3a-4ce1cf5a2d79.png)
![Screenshot (1650)](https://user-images.githubusercontent.com/56502606/121810488-b9cb8300-cc7e-11eb-98b1-df6e3309b586.png)
![Screenshot (1463)](https://user-images.githubusercontent.com/56502606/121810820-e0d68480-cc7f-11eb-90ed-ef6806574190.png)
![Screenshot (1651)](https://user-images.githubusercontent.com/56502606/121810495-c18b2780-cc7e-11eb-897c-22661ac9d579.png)
![Screenshot (1652)](https://user-images.githubusercontent.com/56502606/121810527-dcf63280-cc7e-11eb-91ac-a03e59e80d0c.png)
![Screenshot (1654)](https://user-images.githubusercontent.com/56502606/121810539-e7b0c780-cc7e-11eb-864a-2c44cc3ac965.png)


 
