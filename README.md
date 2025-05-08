# Diabetes Prediction Using Machine Learning
# by Devras Jadhav

---

## Introduction

This project focuses on predicting the likelihood of diabetes in individuals based on various medical features. I used a dataset containing health indicators like glucose, BMI, insulin levels, age, and more. The goal was to apply classification algorithms to predict diabetes presence (binary outcome).

## Objective

The aim was to build and compare multiple machine learning models to determine which one most effectively predicts diabetes. The models include Logistic Regression, Decision Tree, Random Forest, Support Vector Classifier, and Neural Network. I also used PCA for dimensionality reduction.

## Dataset

The dataset used (`diabetes.csv`) consists of 768 rows and 9 columns:
- **Pregnancies**
- **Glucose**
- **BloodPressure**
- **SkinThickness**
- **Insulin**
- **BMI**
- **DiabetesPedigreeFunction**
- **Age**
- **Outcome** (Target: 0 = No Diabetes, 1 = Diabetes)

## Data Cleaning and Preprocessing

- Replaced all invalid zero entries in Glucose, BloodPressure, Insulin, SkinThickness, and BMI with NaNs.
- Imputed missing values with the **median** to preserve distribution.
- Removed duplicate rows (none found).
- Confirmed the final dataset had 768 clean records.

## EDA & Visualization

- Histograms, boxplots, and heatmaps were created to understand data distribution and outliers.
- Outcome distribution was slightly imbalanced (500+ no diabetes, 200+ with diabetes).
- Plots showed Glucose and BMI to be strong predictors.

## Train-Test Split & Feature Scaling

While working on this project, I came to understand the deeper implications of how data preprocessing affects model performance. Initially, my approach was quite straightforward—I cleaned the data, handled missing values, and scaled the features before splitting the dataset. But this turned out to be a mistake that could have compromised the entire project. The reason is something known as data leakage. When transformations like scaling are applied to the whole dataset before splitting it into training and testing sets, it unintentionally allows information from the test set to influence the training process. This leads to overly optimistic performance metrics and models that fail to generalize well to unseen data.

To fix this, I reordered my pipeline. The first major change was using `train_test_split()` from scikit-learn to divide the dataset into training and testing sets **before** doing any scaling. Only after the data was split did I apply `StandardScaler()` to the training set and then used the same scaler to transform the test set. This ensured that the test data remained truly unseen by the model during training.

But it wasn't just about where to scale—it was also about which models needed scaling. Through this project, I learned that not every machine learning model reacts the same way to unscaled data. Algorithms like **Logistic Regression**, **Support Vector Machines (SVC)**, and **Neural Networks (MLPClassifier)** are sensitive to the scale of input features. These models operate on distances, dot products, or gradient-based updates, all of which are directly impacted by the range of feature values. For these models, I made sure to use the scaled version of the training and testing sets.

On the other hand, models like **Decision Tree** and **Random Forest** don’t care about feature scale. These algorithms rely on hierarchical threshold-based splitting of the data. Whether a feature is measured in grams or kilometers doesn't affect the model’s logic; it just finds a threshold that best splits the data. Therefore, I passed unscaled data to these models. This not only avoided unnecessary computations but also ensured their performance remained interpretable and stable.

Another important element of this project was visualizing and understanding the **feature importance**. For the Random Forest model, I used the `.feature_importances_` attribute to determine which features were contributing most to the predictions. Glucose was by far the most influential variable, followed by BMI and Age. To visualize this, I plotted a horizontal bar chart, which gave a clear representation of the importance ranking. This step was not just informative but also critical in reinforcing my understanding of which variables truly matter in diabetes prediction.

As I was building each model, I paid special attention to the preprocessing steps being applied. For instance, when implementing Logistic Regression, I applied `StandardScaler()` only on the `X_train` and then transformed `X_test` using the same fitted scaler. I kept the scaled versions of the training and test data separate from the unscaled ones to avoid any mix-up. For models like Decision Tree and Random Forest, I skipped the scaling step entirely and used the raw input features directly.

Here's how the logic was structured in code:

```python
# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling only for models that require it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Tree models use unscaled data
X_train_tree = X_train
X_test_tree = X_test
```

I also made sure that the evaluation metrics I was using—accuracy, precision, recall, and F1-score—were consistent across all models. This made comparison meaningful and ensured I wasn’t favoring any model just because of preprocessing differences. Every model was trained on the exact same target values and test indices, with only their input features being scaled or not depending on the model’s requirement.

I faced some difficulties initially when I realized that applying the scaler to the entire dataset was giving suspiciously high accuracy scores. Upon digging deeper, I realized this was a classic symptom of data leakage. Fixing this meant revisiting multiple parts of the pipeline, re-running training processes, and re-evaluating the results from scratch. Though it took extra time, the end result was a cleaner, more correct project with results I could trust.

Another issue I had to resolve was ensuring consistency across model performance reporting. Since I had both scaled and unscaled versions of the data, I needed to make sure each model used the correct version during both training and evaluation. Mixing them up would not only confuse the outcome but could also silently introduce errors that are hard to catch. Keeping a clear structure in my notebook—naming datasets properly and keeping cell outputs organized—helped avoid this problem.

What I found particularly insightful was how different models behave depending on whether the data is scaled or not. For example, my Logistic Regression model performed significantly better after proper scaling was applied. The training process converged faster and the classification report showed improvements in both precision and recall. In contrast, the performance of Random Forest remained virtually the same whether the data was scaled or not, reinforcing what I had read about scale-invariant algorithms.

In conclusion, this part of the project turned out to be the most technically revealing for me. It demonstrated how subtle mistakes in preprocessing can lead to large consequences in model performance. More importantly, it taught me the correct methodology: always split the data first, apply preprocessing only to training data, and scale only where necessary. It also showed me the power of model interpretability tools like feature importance plots, which not only help in model evaluation but also give deeper insights into the problem domain. These lessons will stick with me for all future machine learning work I do.. Initially, I had applied scaling to the entire dataset before performing a train-test split. That seemed fine at first, but I eventually realized this causes data leakage. When the model has access to information from the test set—even indirectly through scaling—it can inflate accuracy results and lead to poor generalization.

To address this, I first performed the train-test split using `train_test_split()`. Once I had separated the data into training and testing sets, I applied scaling using `StandardScaler()`—but only for the models that actually needed it. Logistic Regression, Support Vector Classifier (SVC), and Neural Networks are models that depend heavily on feature scale. Their performance improves when all features are normalized to similar ranges, so I scaled only for those models.

For models like Decision Tree and Random Forest, I made a conscious choice not to scale the data at all. These models are based on threshold splitting and are inherently insensitive to the magnitude of the input features. In fact, scaling them can introduce noise or unnecessary complexity. Instead, I passed the raw unscaled data directly to these models, which kept their interpretation straightforward and their performance stable.

Another essential part of this process was understanding the value of model explainability. For the Random Forest classifier, I calculated and visualized the feature importances. This allowed me to identify which features had the most influence on the predictions. Unsurprisingly, Glucose level came out as the most important, followed by BMI and Age. Plotting these helped not only in evaluating the model but also in understanding the underlying patterns in the data.

These adjustments—reordering my scaling logic, selectively applying it to only the right models, and visualizing feature importance—significantly improved the integrity and accuracy of my final results. More importantly, I now have a much stronger understanding of why these steps matter and how they fit into a proper machine learning pipeline.. One of the most important changes I made was making sure that scaling of the data happens only after performing the train-test split. Initially, I had done the scaling before splitting, which could have led to data leakage—this means information from the test set could unintentionally influence the training set, which ruins the integrity of the model.

After learning this, I updated my pipeline so that the dataset is first split using `train_test_split()`. Then, I applied `StandardScaler()` only on the training and testing data that required it. Specifically, I applied scaling only to Logistic Regression and SVC models. These models are sensitive to the scale of the input features, and scaling improves their performance and convergence.

On the other hand, I made sure that tree-based models like Decision Tree and Random Forest were trained on unscaled data. These models do not require feature scaling because they split nodes based on feature thresholds rather than distances. Applying scaling here would not help and could even complicate interpretability.

Additionally, I extracted and visualized the feature importances from the Random Forest model. This gave a clear view of which features had the most impact in predicting diabetes. Glucose, BMI, and Age were among the top contributors. Displaying feature importance not only strengthened the model evaluation but also improved my understanding of the dataset.

These steps—careful ordering of preprocessing, applying scaling selectively, and using interpretability tools—were crucial in making my workflow both correct and robust.. I also applied scaling **only to models that require it**—namely, Logistic Regression and SVC.
- I performed **train-test split first**, then applied **scaling only to Logistic Regression and SVC** models to prevent data leakage.
- **Tree-based models** (Decision Tree and Random Forest) were trained on **unscaled data**.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling for models that require it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Unscaled for tree models
X_train_tree = X_train
X_test_tree = X_test
```

## PCA – Principal Component Analysis

- Applied PCA on **scaled training data only** to avoid data leakage.
- Visualized the first two principal components.
- PCA helped visualize class separation though not used in final model.

## Models Used

### Model 1: Logistic Regression (on scaled data)
- GridSearchCV was used to tune `C` parameter.
- Accuracy approx **75 percent**.
- Scaling done **after split**, as per remark.

### Model 2: Decision Tree (on unscaled data)
- No scaling applied.
- Accuracy approx **75 percent**.

### Model 3: Random Forest (on unscaled data)
- GridSearchCV used for `n_estimators` and `max_depth`.
- Accuracy: **77 percent**
- **Displayed feature importances** using `feature_importances_`.

```python
importances = rf.feature_importances_
pd.Series(importances, index=X.columns).sort_values(ascending=False).plot.bar()
```

### Model 4: Support Vector Classifier (on scaled data)
- GridSearch used to tune `C` and `gamma`.
- Scaling done properly after train-test split.
- Accuracy: **76 percent**

### Model 5: Neural Network (MLPClassifier)
- Applied on scaled data.
- Accuracy around **76 percent**
- Used one hidden layer with `relu` and `adam` optimizer.

## Model Comparison

| Model              | Accuracy |
|-------------------|----------|
| Logistic Regression | 75%     |
| Decision Tree       | 75%     |
| Random Forest       | 77%     |
| SVC                 | 76%     |
| Neural Network      | 76%     |

**Random Forest performed the best**, and feature importances confirmed Glucose, BMI, and Age as top predictors.



## Challenges Faced

- Understanding data leakage and correcting early mistakes.
- Avoiding scaling mistakes before train-test split.
- Implementing GridSearchCV efficiently.
- Cleaning and imputing zero values appropriately.

## Learnings and Experience

- I learned the importance of proper data preprocessing and avoiding leakage.
- I now understand the strengths and weaknesses of each model better.
- I gained experience in feature scaling, hyperparameter tuning, and visualization.
- Completing this alone helped reinforce every ML pipeline step.

## Conclusion

Random Forest emerged as the best performer. Following all ML best practices and professor feedback helped ensure this final project was technically sound and properly executed.
