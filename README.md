# Diabetes-Prediction-Using-Machine-Learning-Algorithms

## Methodology

### Overview Procedure from Data Collection to Model Deployment: </br>

![flowchart](/images/overview_procedure.png)

### Data Visualization

`Data visualisation` is a useful tool that enables people to grasp and comprehend complex data more easily. It entails presenting data in a graphical or visual style, which can aid in identifying patterns, trends, and relationships that may not be obvious when looking at raw data.

`Skewness of Attribute Distribution`</br>

![Attribute Distribution](/images/attribute_distribution.png)

`Histogram for each attribute in dataset` </br>

![Attributes Histogram](/images/attributes_histogram.png)

`Bar Graph for Distribution of Diabetic` </br>

![](/images/diabetic_distribution.png)

`Box and Whisker plot` </br>

![Diabetic Distribution](/images/box_whisker.png)

### Data Preprocessing

`Data preprocessing` is a method that is used to enhance the data's quality prior to apply mining and ensuring that the data will produce high quality mining results.
Preprocessing of data includes:
* Data Cleaning
* Data Integrating
* Data Transforming
* Reducing Data

#### Pearson Correlation 

`Pearson correlation` is a statistical metric used to quantify the linear relationship between two continuous variables. The Pearson correlation coefficient is a number between -1 and 1, with 1 indicating a perfect positive linear link, -1 indicating a perfect negative linear relationship, and 0 indicating no linear relationship.

`Correlation Matrix for each feature` </br>

![Pearson Correlation Feature](/images/pearson_correlation_feature.png)

`Correlation Matrix between features and outcome` </br>

![Pearson Correlation Label](/images/pearson_correlation_outcome.png)

From the correlation matrix above, we can observe the correlation coefficient of ‘PatientID’ is smaller. It can be considered to remove it from datasets if we want to save the time consume and save the computation cost during model training.

#### StandardScaler 

`StandardScaler` is a preprocessing tool for standardising the features in a dataset. This is important because most of the machine learning algorithms assume that the input data is normally distributed and that all characteristics are of equal magnitude. StandardScaler normally will be applied logistics regression and support vector machine, PCA and so on during training process.

`StandardScaler Data` </br>

![StandardScaler Data](/images/standard_scaler.png)

### Model Traning With Wrapper Method and Hyperparameters Tuning

`Wrapper method` is one of the feature selection techniques that is a strategy for selecting the most important features to include in a model. This strategy entails evaluating several features combinations and selecting the one that results in the best model performance. 

`Hyperparameter tuning` is the process of determining the best settings for machine learning. Hyperparameters that are not learned from data but can be adjusted or used as defaults before training the model. The model can be optimised by tuning the hyperparameters to achieve the best feasible performance on a given dataset.

`Weightage of Training, Validiation and Test Set` </br>

![Data Weightage](/images/data_weightage.png)

#### Logistic Regression

Feature Selection with SFS: </br>
![Logistics Regression SFS](/images/logistic_sfs.png)

Hyperparameters tuning in Logistic Regression with Grid Search:
| Hyperparameter | Values |
| -------------  | ------ |
|    max_iter    | 10000  |
|       C        |  0.059 |
|    penalty     |   l2   |
|    solver      | lbfgs  |

#### K-Nearest Neighbors

Feature Selection with SFS: </br>
![K Nearest Neighbors SFS](/images/knn_sfs.png)

Hyperparameters tuning in K Nearest Neighbors with graph method:
![K Nearest Neighbors SFS](/images/knn_hp.png)

#### Decision Tree 

Feature Selection with SFS: </br>
![Decision Tree SFS](/images/decision_tree_sfs.png)

Hyperparameters tuning in Decision Tree with Grid Search:
| Hyperparameter | Values |
| -------------  | ------ |
|    max_depth   | 10  |
| max_leaf_nodes | 40  |

#### Support Vector Machine

Feature Selection with SFS: </br>
![Support Vector Machine SFS](/images/svm_sfs.png)

Hyperparameters tuning in Support Vector Machine with Random Search:
| Hyperparameter | Values |
| -------------  | ------ |
|    max_depth   | 40  |
| max_features | auto  |
|    n_estimators   | 100  |
| min_samples_leaf | 1  |
| min_samples_split | 1  |

#### Random Forests 

Feature Selection with SFS: </br>
![Random Forest SFS](/images/rf_sfs.png)

Hyperparameters tuning in Random Forests with Grid Search:
| Hyperparameter | Values |
| -------------  | ------ |
|    gamma    | 0.0001  |
|       C        |  1000 |
|    kernel     |   rbf   |
|    probability      | True  |

#### XGBoost

Feature Selection with SFS: </br>
![XGBoost SFS](/images/xgboost_sfs.png)

Hyperparameters tuning in XGBoost with Random Search:
| Hyperparameter | Values |
| -------------  | ------ |
|    max_depth   | 40  |
| subsample | 0.7  |
|    min_child_weight   | 1  |
| learning_rate | 0.2  |
| gamma | 0.2  |
| colsample_bytree | 0.2  |

#### Light-GBM 

Feature Selection with SFS: </br>
![Light-GBM SFS](/images/lgbm_sfs.png)

Hyperparameters tuning in Light-GBM with Random Search:
| Hyperparameter | Values |
| -------------  | ------ |
|    max_depth   | 15  |
| num_leaves | 30  |
|    n_estimators   | 100  |
| subsample | 0.8  |
| colsample_bytree | 0.1  |



