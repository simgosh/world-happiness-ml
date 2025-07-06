# Supervised Machine Learning on World Happiness Report (2019)
### Happiness Score Prediction and Classification Using Machine Learning Techniques

****The dataset was downloaded from Kaggle.

This project was developed as part of the Supervised Learning Final Assignment for the Graduate Program in Data Science at Ege University. The goal is to apply various supervised machine learning algorithms to the 2019 World Happiness Report dataset, tackling both regression and classification problems.

### The dataset contains happiness scores (Score) for countries along with key economic, social, and political indicators. The analysis is divided into two main tasks:
1) Regression Task
Objective: Predict the continuous happiness score using various models.
Algorithms Applied:

Linear Regression (LM)
Regression Tree (RT)
Bagging Regression Tree (BRT)
Random Forest Regression (RFR)

Performance Summary (on Test Set):
Best-performing model: Bagging Regression (BRT)
Worst performance: Linear Regression (negative R², indicating a poor fit)


2) Classification Task
Objective: Classify countries into two groups: high or low happiness.
Threshold: Median happiness score used to create a balanced binary target variable.
Algorithms Applied:

Classification Tree (CT)
Bagging Classification Tree (BCT)
Random Forest Classifier (RFC)
Logistic Regression (LR)
Linear Discriminant Analysis (LDA)
Quadratic Discriminant Analysis (QDA)

Model Evaluation:
Models were compared based on accuracy, precision, recall, F1-score, and ROC AUC.
Best overall performance: Bagging Classifier
Weakest performance: Logistic Regression and QDA

## Conclusions & Insights :
- Bagging-based models (in both regression and classification) outperformed single estimators in terms of accuracy and robustness.
- Linear models (Linear and Logistic Regression) underperformed due to their inability to capture non-linear relationships in the data.
- Feature importance analysis revealed that variables like GDP per capita, Social support, and Healthy life expectancy are the most influential.
- Model performances were evaluated on both training and test datasets, with careful attention to overfitting.
- Using the median threshold ensured balanced class distribution and improved classification outcomes.


![histogram](https://github.com/user-attachments/assets/fc87826e-e1a8-43c5-aa27-2d529c80824d)
The histogram of the 'Score' feature shows that happiness scores across countries are mostly concentrated between 6 and 7.5, and the distribution is slightly right-skewed.

![double plot](https://github.com/user-attachments/assets/41858776-f10e-423a-b572-62433ab64a52)
GDP per Capita and Social Support are two features that have a strong positive correlation with the 'Score'. In particular, as GDP per Capita increases, the happiness score tends to increase significantly.

![top10](https://github.com/user-attachments/assets/cbeeb32e-87d9-410a-a145-6f66a8fabb24)
The graph on the left shows the top 10 happiest countries in the world, with Finland being the happiest country.

![Rplot](https://github.com/user-attachments/assets/28e1aa62-5144-438e-b8ae-4efab6410099)
Finally, interpreting the results visually, the Bagging model has the lowest error values in terms of both RMSE and MAE, indicating it provides the most accurate predictions. Looking at the R² values, Bagging also explains the variance in the data best. While the Regression Tree performs well, the Random Forest and Linear Regression models show higher error values and lower R² scores, performing weaker compared to the other models. These results demonstrate that ensemble methods, especially Bagging, are effective in improving prediction accuracy on this dataset.



