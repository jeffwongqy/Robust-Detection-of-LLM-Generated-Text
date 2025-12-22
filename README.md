# Robust Methods for Detecting LLM Generated Text

## Aim: 
To develop a robust system for acurrately detecting LLM-generated text by applying reliable machine learning techniques and maintaining a reproducible, automated development workflow usign Git and Jenkins. 



## Objectives:
- Design and implement models capable of distinguishing human-written and LLM-generated text with high accuracy. 
- Build a well-versioned codebase using Git to support collaboration, traceability, and controlled experimentation. 
- Set up Jenkins pipelines to automate testing, model evaluation, and deployment. 
- Ensure robustness by validating the system across diverse datasets and different LLM outputs. 

## Preprocessing Steps: 
1. Data Loading & Cleaning:
The dataset is loaded from a CSV file using Pandas. Irrelevant columns such as id and prompt_id are removed, and duplicate rows are checked to ensure data quality. 

2. Exploratory Text Analysis: 
Basic linguistic features are computed from the raw text, including word count, average sentence length, and vocabulary diversity. These features help analyze structural difference between human-written and LLM-generated text. Their distriutions are visualized for insight. 

<img src = "Answer Length Distribution (Words).png" width = "500">

<img src = "Average Sentence Length Distribution.png" width = "500">

<img src = "Vocabulary Diversity Distribution.png" width = "500">

3. Text Cleaning:
Each text is normalised by converting to lowercase, removing digits, punctuation, and special characters. This reduces noise and ensures consistent input for feature extraction. 

4. Feature Extraction (TF-IDF):
Cleaned text is transformed into numerical features using TF-IDF with unigrams and bigrams. This captures word importance and contextual patterns useful for classification. 

5. Class Distribution Analysis:
The class labels (human vs generated) are visualized to identify imblance in the dataset.

<img src = "Class Distribution (Before Resampling).png" width = "500">

6. Resampling:
SMOTE combined with TOmek links is applied to balance the dataset by oversampling the minority class and cleaning overlapping samples, improving model robustness. 

<img src = "Class Distribution (After Resampling).png" width = "500">

7. Train-Test Split:
The balanced data is split into training and testing sets to allow fair evaluation. 

8. Saving Preprocessed Data:
The final processed datasets are saved using joblib for reuse in model training and deployment pipelines. 

## Model Training: 
Three supervised classifiers are trained for comparison: Logistic Regression, Decision Tree, and Random Forest. This allows evaluation of both linear and tree-based approaches for LLM text detection. 

For each model, a predefined hyperparameter grid is used. GridSearchCV with 5-fold Straitified Cross-Validation ensures optimal parameter selection while preserving class balance in each fold. 

Each model is trained on the training data using accuracy as the evaluation metric. The best-performing hyperparameter configuration and cross-validation score are reported. 

The trained and optimized models are saved as .pkl files using joblib for later evaluation, deployment, or CI/ CD integration with Jenkins. 

## Model Evaluation:
Each model generates class predictions and class probabilities on the test set. Probabilities are required for ROC-AUC analysis. 

Precision, recall, F1-score, and accuracy are computed for both classes. All models achieve perfect scores (1.00) across all metrics, indicating flawless classification performance on the test data. 


Classification Report for Logistic Regression:
<img src = "logreg_classification_report.png" width = "500">


Classification Report for Decision Tree Classifier: 
<img src = "dtc_classification_report.png" width = "500">


Classification Report for Random Forest Classifier:
<img src = "rfc_classification_report.png" width = "500">

Confusion matrices show zero misclassification, with all samples correctly classified into their respective classes. This confirms the absence of false positives and false negatives. 


Confusion Matrix for Logistic Regression:
<img src = "Confusion Matrix for Logistic Regression.png" width = "500">


Confusion Matrix for Decision Tree Classifier:
<img src = "Confusion Matrix for Decision Tree Classifier.png" width = "500">


Confusion Matrix for Random Forest Classifier:
<img src = "Confusion Matrix for Random Forest Classifier.png" width = "500">

ROC curves for all models closely follow the top left corner, with an AUC score of 1.00. This demonstrates perfect stability between the human-written and AI-generated text. 


ROC-AUC Curve for Logistic Regression:
<img src = "ROC-AUC Curve for Logistic Regression.png" width = "500">


ROC-AUC Curve for Decision Tree Classifier:
<img src = "ROC-AUC Curve for Decision Tree Classifier.png" width = "500">


ROC-AUC Curve for Random Forest Classifier:
<img src = "ROC-AUC Curve for Random Forest Classifier.png" width = "500">

The consistently perfect results across confusion matrices, classification reports, and ROC-AUC scores indicate that all three models perform exceptionally well on the given dataset. 

## Tools
- Python 
- Jenkins
- Git /GitHub
- Docker 

## References
Mo, Y., Qin, H., Dong, Y., Zhu, Z., & Li, Z. (2024, April 6). Large Language Model (LLM) AI text generation detection based on transformer deep learning algorithm. arXiv.org. https://arxiv.org/abs/2405.06652


