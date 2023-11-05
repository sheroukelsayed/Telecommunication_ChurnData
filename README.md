# Telecommunication_ChurnData
 create a model for a telecommunication company, to predict when its customers will leave for a competitor, so that they can take some action to retain the customers.
we compare 5 popular machine learning algorithms:KNN, Decision Tree, Logistic Regression, Random Forest, and SVM
​Here is a comparison of the 5 machine learning algorithms with explanations merged with key equations:

K-Nearest Neighbors (KNN):
KNN makes predictions by searching the training set for the K most similar instances (neighbors) and outputting the majority class among the neighbors. It does not have an explicit mathematical model, and instead relies on similarity measures:

Prediction = majority vote of K closest training samples 

Decision Tree:
Decision trees work by recursively splitting the data on feature values that result in the largest information gain at each node. Information gain is based on reducing entropy in the split data:

Information Gain = Entropy(parent) - Σ[(Probability of split) x Entropy(child)]

At each node, choose the feature that maximizes information gain. Predictions are made by following decisions from root to leaf node.

Logistic Regression:
Logistic regression calculates the probability P(Y=1|X) using the logistic function:

P(Y=1|X) = 1/(1+e^(-wx+b))

It optimizes the weights w and bias b to maximize the likelihood of the training data. The decision boundary is formed by thresholding the probability.

Random Forest:
Random forest creates multiple decision trees on random subsets of data and features. Each tree makes a prediction and the final prediction is the majority vote:

Prediction = mode of predictions from all decision trees

By aggregating predictions across diverse trees, overfitting is reduced.

Support Vector Machine (SVM):
SVM constructs a hyperplane f(x) = w^Tx + b to separate the classes with maximum margin. For non-linearly separable classes, it maps data to a high-dim space using kernels. The prediction is made by determining which side of the hyperplane a data point falls on. The optimization objective is to maximize the margin between classes.


About the dataset
We will use a telecommunications dataset for predicting customer churn. This is a historical customer dataset where each row represents one customer. The data is relatively easy to understand, and you may uncover insights you can use immediately. Typically it is less expensive to keep customers than acquire new ones, so the focus of this analysis is to predict the customers who will stay with the company.
This data set provides information to help you predict what behavior will help you to retain customers. You can analyze all relevant customer data and develop focused customer retention programs.

The dataset includes information about:

Customers who left within the last month – the column is called Churn
Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
Customer account information – how long they had been a customer, contract, payment method, paperless billing, monthly charges, and total charges
Demographic info about customers – gender, age range, and if they have partners and dependents
Load the Telco Churn data
Telco Churn is a hypothetical data file that concerns a telecommunications company's efforts to reduce turnover in its customer base. Each case corresponds to a separate customer and it records various demographic and service usage information. Before you can work with the data, you must use the URL to get the ChurnData.csv.

To download the data, you can use !wget to download it from IBM Object Storage.
#Click here and press Shift+Enter
path= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
