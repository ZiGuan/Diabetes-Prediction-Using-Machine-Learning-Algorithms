# Diabetes-Prediction-Using-Machine-Learning-Algorithms

## Methodology

### Overview Procedure from Data Collection to Model Deployment

![flowchart](/images/overview_procedure)

### Data Visualization

Data visualisation is a useful tool that enables people to grasp and comprehend complex data more easily. It entails presenting data in a graphical or visual style, which can aid in identifying patterns, trends, and relationships that may not be obvious when looking at raw data.

Skewness of Attribute Distribution

![Attribute Distribution](/images/attribute_distribution)

Histogram for each attribute in dataset

![](/images/attributes_histogram)

Bar graph for distribution of diabetic

![](/images/diabetic_distribution)

Box and Whisker plot

![](/images/box_whisker)

### Data Preprocessing

Data preprocessing is a method that is used to enhance the data's quality prior to apply mining and ensuring that the data will produce high quality mining results.
Preprocessing of data includes:
* Data Cleaning
* Data Integrating
* Data Transforming
* Reducing Data

#### Pearson Correlation 

Pearson correlation is a statistical metric used to quantify the linear relationship between two continuous variables. The Pearson correlation coefficient is a number between -1 and 1, with 1 indicating a perfect positive linear link, -1 indicating a perfect negative linear relationship, and 0 indicating no linear relationship.

Correlation matrix for each feature

![](/images/pearson_correlation_feature)

Correlation matrix between features and outcome

![](/images/pearson_correlation_outcome)

From the correlation matrix above, we can observe the correlation coefficient of ‘PatientID’ is smaller. It can be considered to remove it from datasets if we want to save the time consume and save the computation cost during model training.

#### StandardScaler 

StandardScaler is a preprocessing tool for standardising the features in a dataset. This is important because most of the machine learning algorithms assume that the input data is normally distributed and that all characteristics are of equal magnitude. StandardScaler normally will be applied logistics regression and support vector machine, PCA and so on during training process.



