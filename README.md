# Diabetes-Prediction-Using-Machine-Learning-Algorithms

Diabetes is a prevalent chronic condition that can seriously endanger human health. When blood glucose levels are greater than normal, diabetes can be diagnosed whether is caused from various kinds of biological factors. Machine learning algorithms can be used to predict diabetes by selecting and processing relevant medical data, training, and validating the model, and incorporating it into clinical decision support systems.

## Model Accuracy

|               | After Feature Selection | After Hyperparameter Tuning  | Final Accuracy (Test Set)
|---------------------| ----------------------- | -------------------------- | --------- |
| Logistic Regression | 79.60 %  | 79.60 % |  78.04 %
| K-Nearest Neighbors| 90.80 % | 91.33 % | 88.76 %
| Decision Tree |  89.60 % | 91.60 % | 90.04 %
| Support Vector Machine | 83.20 % | 85.97 % | 91.24 %
| Random Forest | 94.13 % | 94.40 % | 94.20 %
| XGBoost | 95.87 % | 96.40 % | 96.52 %
| Light-GBM | 94.93 % | 95.60 % | 96.64 %

## Precision-Recall Curve
![PR Curve](/images/pr_curve.png)

## Confusion Matrix & Classification Report

### Logistics Regression                                               
|               | Predicted Negative | Predicted Positive |  
|---------------| ------- | ------ |
|Actual Negative| 1464 | 198 |
|Actual Positive| 351 | 487 |

|Performance Measures| Precision | Recall | F1-Score | Support
|---------------| ------- | ------ | ------- | -------
|0              | 0.8066	| 0.8809	| 0.8421	| 1662
|1| 0.7109	| 0.5811	| 0.6395	| 838
|Accuracy| 0.7804	| 0.7804	| 0.7804	| 0.7804
|Macro Avg| 0.7588	| 0.7310	| 0.7408	| 2500
|Weighted Avg| 0.7745	| 0.7804	| 0.7742	| 2500

### K-Nearest Neighbors 
|               | Predicted Negative | Predicted Positive |
|---------------| -------  | ------ |
|Actual Negative| 1578 | 84 |
|Actual Positive| 164 | 674 |

|Performance Measures| Precision | Recall | F1-Score | Support
|---------------| ------- | ------ | ------- | -------
|0              | 0.9059	| 0.9495	| 0.9271	| 1662
|1| 0.8892	| 0.8043	| 0.8446	| 838
|Accuracy | 0.9008	| 0.9008	| 0.9008	| 0.9008
|Macro Avg | 0.8975	| 0.8769	| 0.8859	| 2500
|Weighted Avg| 0.9003	| 0.9008	| 0.8995	| 2500

### Decision Tree
|               | Predicted Negative | Predicted Positive |
|---------------| ------  | ------ |
|Actual Negative| 1572 | 90 |
|Actual Positive| 129 | 709 |

|Performance Measures| Precision | Recall | F1-Score | Support
|---------------| ------- | ------ | ------- | -------
|0              | 0.9242	| 0.9458 | 0.9349	| 1662
|1|  0.8874	|  0.8461	|  0.86626	| 838
|Accuracy | 0.9124	|  0.9124	|  0.9124	| 0.9124
|Macro Avg | 0.9058	| 0.8960	| 0.9005	| 2500
|Weighted Avg| 0.9118	| 0.9124	| 0.91195	| 2500

### Support Vector Machine
|               | Predicted Negative | Predicted Positive |
|---------------| ------ | ------ |
|Actual Negative| 1546 | 116 |
|Actual Positive| 165 | 673 |

|Performance Measures| Precision | Recall | F1-Score | Support
|---------------| ------- | ------ | ------- | -------
|0              | 0.9036	| 0.9302 |  0.9167	| 1662
|1| 0.8530	|  0.8031	 |  0.8273	| 838
|Accuracy | 0.8876	|  0.8876	|  0.8876	|  0.8876
|Macro Avg | 0.8783	|  0.8667	| 0.8720	| 2500
|Weighted Avg| 0.8866	| 0.8876 | 0.8867	| 2500

### Random Forest
|               | Predicted Negative | Predicted Positive |
|---------------| -------------  | ------ |
|Actual Negative| 1547 | 56 |
|Actual Positive| 107 | 790 |

|Performance Measures| Precision | Recall | F1-Score | Support
|---------------| ------- | ------ | ------- | -------
|0              | 0.9353	| 0.9651	| 0.9500	| 1662
|1| 0.9338	| 0.8807	| 0.9065	| 838
|Accuracy | 0.9348	| 0.9348	| 0.9348	| 0.9348
|Macro Avg | 0.9346	| 0.9229	| 0.9282	| 2500
|Weighted Avg| 0.9348	| 0.9348	| 0.9344	| 2500

### XGBoost
|               | Predicted Negative | Predicted Positive |
|---------------| -------------  | ------ |
|Actual Negative| 1622 | 40 |
|Actual Positive| 47 | 791 |

|Performance Measures| Precision | Recall | F1-Score | Support
|---------------| ------- | ------ | ------- | -------
|0              | 0.9718	| 0.9759 |	0.9739	| 1662
|1| 0.9519	| 0.9439	| 0.9479	| 838
|Accuracy | 0.9652 |	0.9652	| 0.9652	| 0.9652
|Macro Avg | 0.9619	| 0.9599	| 0.9609	| 2500
|Weighted Avg| 0.9651	| 0.9652	| 0.9652	| 2500

### Light-GBM
|               | Predicted Negative | Predicted Positive |
|---------------| -------------  | ------ |
|Actual Negative| 1629 | 33 |
|Actual Positive| 51 | 787 |


|Performance Measures| Precision | Recall | F1-Score | Support
|---------------| ------- | ------ | ------- | -------
|0              | 0.9696	| 0.9801	| 0.9749	| 1662
|1| 0.9598	| 0.9391	| 0.9493	| 838
|Accuracy | 0.9664	| 0.9664	| 0.9664	| 0.9664
|Macro Avg | 0.9647	| 0.9596	| 0.9621	| 2500
|Weighted Avg| 0.9663	| 0.9664	| 0.9663	| 2500

## Model Deployment
![](/images/deployment_highrisk.png)  ![](/images/deployment_lowrisk.png)

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

### Model Training With Wrapper Method and Hyperparameters Tuning

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




