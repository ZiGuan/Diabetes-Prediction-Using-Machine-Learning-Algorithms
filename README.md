# Diabetes-Prediction-Using-Machine-Learning-Algorithms

## Methodology

### Overview Procedure from Data Collection to Model Deployment: </br>

![flowchart](/images/overview_procedure.png)

### Data Visualization

Data visualisation is a useful tool that enables people to grasp and comprehend complex data more easily. It entails presenting data in a graphical or visual style, which can aid in identifying patterns, trends, and relationships that may not be obvious when looking at raw data.

`Skewness of Attribute Distribution`</br>

![Attribute Distribution](/images/attribute_distribution.png)
<figcaption>Skewness of Attribute Distribution</figcaption>

`Histogram for each attribute in dataset` </br>

![Attributes Histogram](/images/attributes_histogram.png)

`Bar Graph for Distribution of Diabetic` </br>

![](/images/diabetic_distribution.png)

`Box and Whisker plot` </br>

![Diabetic Distribution](/images/box_whisker.png)

### Data Preprocessing

Data preprocessing is a method that is used to enhance the data's quality prior to apply mining and ensuring that the data will produce high quality mining results.
Preprocessing of data includes:
* Data Cleaning
* Data Integrating
* Data Transforming
* Reducing Data

#### Pearson Correlation 

Pearson correlation is a statistical metric used to quantify the linear relationship between two continuous variables. The Pearson correlation coefficient is a number between -1 and 1, with 1 indicating a perfect positive linear link, -1 indicating a perfect negative linear relationship, and 0 indicating no linear relationship.

`Correlation Matrix for each feature` </br>

![Pearson Correlation Feature](/images/pearson_correlation_feature.png)

`Correlation Matrix between features and outcome` </br>

![Pearson Correlation Label](/images/pearson_correlation_outcome.png)

From the correlation matrix above, we can observe the correlation coefficient of ‘PatientID’ is smaller. It can be considered to remove it from datasets if we want to save the time consume and save the computation cost during model training.

#### StandardScaler 

StandardScaler is a preprocessing tool for standardising the features in a dataset. This is important because most of the machine learning algorithms assume that the input data is normally distributed and that all characteristics are of equal magnitude. StandardScaler normally will be applied logistics regression and support vector machine, PCA and so on during training process.

`StandardScaler Data` </br>

![StandardScaler Data](/images/standard_scaler.png)

### MODEL TRAINING WITH WRAPPER METHOD AND HYPERPARAMETER TUNING

`Training Data Distribution` </br>

![Data Weightage](/images/data_weightage.png)





