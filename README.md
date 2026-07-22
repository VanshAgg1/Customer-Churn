# 🚀 Customer Churn Prediction with XGBoost

This repository provides a robust solution for predicting customer churn in a banking dataset using the powerful **XGBoost** algorithm. The project focuses on maximizing the **ROC-AUC score**, a critical metric for imbalanced classification problems, through a meticulously crafted machine learning pipeline that includes extensive data preprocessing and hyperparameter tuning.

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Stars](https://img.shields.io/github/stars/VanshAgg1/Customer-Churn?style=flat&color=yellow)](https://github.com/VanshAgg1/Customer-Churn/stargazers)
[![Forks](https://img.shields.io/github/forks/VanshAgg1/Customer-Churn?style=flat&color=purple)](https://github.com/VanshAgg1/Customer-Churn/network/members)
[![License: No License](https://img.shields.io/badge/License-No%20License-red.svg)](https://github.com/VanshAgg1/Customer-Churn/)

---

## 📝 Table of Contents

*   [🌟 Project Overview](#-project-overview)
*   [✨ Features](#-features)
*   [🛠️ Tech Stack](#%EF%B8%8F-tech-stack)
*   [📦 Installation](#-installation)
*   [🚀 Usage](#-usage)
*   [📊 How to Use (Real-World Application)](#-how-to-use-real-world-application)
*   [📂 Project Structure](#-project-structure)
*   [📚 API Reference](#-api-reference)
*   [🤝 Contributing](#-contributing)
*   [📜 License](#-license)
*   [🔗 Important Links](#-important-links)
*   [udos Footer](#udos-footer)

---

## 🌟 Project Overview

Customer churn is a significant challenge for banks, impacting revenue and growth. This project addresses this by developing a predictive model to identify customers at high risk of churning (leaving the bank). Leveraging the **XGBoost** algorithm, a highly efficient and flexible gradient boosting framework, the solution aims to provide early warnings, enabling proactive retention strategies.

Developed in the context of a Kaggle challenge, the primary objective is to achieve a high **ROC-AUC score**, indicating excellent discrimination ability between churning and non-churning customers. The pipeline incorporates robust data preprocessing techniques, including scaling numerical features and one-hot encoding categorical features, followed by an optimized XGBoost classifier. Hyperparameter tuning using `RandomizedSearchCV` ensures the model's performance is maximized on unseen data.

### Dataset

The model is trained on a dataset comprising various customer attributes:

*   **Target Variable**: `Exited` (binary: `1` if the customer churned, `0` otherwise).
*   **Numerical Features**: `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`.
*   **Categorical Features**: `Geography`, `Gender`.

Both training and test datasets are expected to be available (`train.csv`, `test.csv`) to develop the model and generate predictions for submission.

---

## ✨ Features

This project comes with several key features designed for robust churn prediction:

*   **Advanced Machine Learning Model**: Utilizes **XGBoost Classifier** for high-performance and accurate predictions. 🌳
*   **Comprehensive Data Preprocessing**: Implements `StandardScaler` for numerical features and `OneHotEncoder` for categorical features to prepare data for modeling. ⚙️
*   **Pipeline Integration**: A streamlined `sklearn.pipeline.Pipeline` integrates preprocessing and the XGBoost model for efficient workflow and reproducibility. 🏗️
*   **Hyperparameter Optimization**: Employs `RandomizedSearchCV` to fine-tune XGBoost hyperparameters, focusing on maximizing the ROC-AUC score. The current configuration actively tests specific optimized parameters:
    *   `n_estimators`: `200`
    *   `max_depth`: `3`
    *   `learning_rate`: `0.05`
    *   `subsample`: `0.8`
    *   `colsample_bytree`: `0.8`
    *   `gamma`: `0.1`
    *   `min_child_weight`: `3`
*   **Stratified Data Splitting**: Ensures the training and validation sets maintain the original distribution of the target variable (`Exited`) to prevent bias. 🎯
*   **ROC-AUC Focused Evaluation**: Model performance is rigorously evaluated using the ROC-AUC score, a standard metric for binary classification, especially with imbalanced datasets. 📈
*   **Automated Submission Generation**: The `main.py` script automatically generates a `submission.csv` file formatted for Kaggle competitions, containing `id` and predicted churn probabilities. 📄

---

## 🛠️ Tech Stack

The following technologies and libraries are used in this project:

*   **Language**: `Python 3.x` 🐍
*   **Data Manipulation**: `pandas`, `numpy`
*   **Machine Learning**: `scikit-learn` (for preprocessing, model selection, evaluation, pipelines)
*   **Gradient Boosting**: `XGBoost`

---

## 📦 Installation

To get started with this project, ensure you have Python 3.x installed. Then, follow these steps to set up your environment:

1.  **Clone the repository**: 
    ```bash
git clone https://github.com/VanshAgg1/Customer-Churn.git
cd Customer-Churn
    ```

2.  **Install the required libraries**: 
    ```bash
pip install pandas numpy scikit-learn xgboost
    ```

---

## 🚀 Usage

To replicate the model training, evaluation, and prediction generation, follow these instructions:

1.  **Download the dataset**: Obtain `train.csv` and `test.csv` from the relevant Kaggle competition (e.g., [Kaggle Bank Customer Churn Prediction](https://www.kaggle.com/competitions/bank-customer-churn-prediction/data) or similar). Place these files in the root directory of the cloned repository.

2.  **Run the main script**: Execute the `main.py` file to train the model and generate the submission file.
    ```bash
python main.py
    ```

3.  **Review Output**: The script will print the best hyperparameters found by `RandomizedSearchCV` (though with `n_iter=1` and specific `param_grid` values, these are the pre-selected parameters) and the achieved ROC-AUC score on the validation set. A file named `submission.csv` will be created in the repository's root directory.

    The output will resemble:
    ```
Best Parameters: {'classifier__subsample': 0.8, 'classifier__n_estimators': 200, 'classifier__min_child_weight': 3, 'classifier__max_depth': 3, 'classifier__learning_rate': 0.05, 'classifier__gamma': 0.1, 'classifier__colsample_bytree': 0.8}
Best AUC-ROC Score: 0.9335...
Submission saved to: submission.csv
    ```

---

## 📊 How to Use (Real-World Application)

This project provides a ready-to-use pipeline for customer churn prediction, directly applicable to banking and other service industries. Here's how it translates to real-world use cases:

1.  **Proactive Customer Retention**: By identifying customers with a high probability of churning (based on the `Exited` probability in `submission.csv`), banks can launch targeted retention campaigns, offering incentives, personalized services, or addressing specific pain points before the customer leaves.

2.  **Resource Allocation**: Optimize marketing and customer service resources by focusing efforts on at-risk customers, rather than a broad, untargeted approach.

3.  **Risk Assessment**: Integrate churn probabilities into a broader risk management framework to understand potential customer base erosion over time.

4.  **Feature Importance Analysis**: While not explicitly implemented in `main.py`, the trained XGBoost model can be further analyzed to determine which features (e.g., Credit Score, Age, Balance) contribute most to churn, providing actionable insights for business strategy.

To adapt this project for a new dataset or continuous monitoring:

*   Replace `train.csv` and `test.csv` with your own customer data, ensuring the column names match those expected by `main.py` (`Geography`, `Gender`, `CreditScore`, etc., and the target `Exited`).
*   Consider re-running the `RandomizedSearchCV` with a broader `param_grid` and `n_iter` if you want to perform a more extensive hyperparameter search tailored to your specific data.

---

## 📂 Project Structure

The repository is structured as follows:

```
Customer-Churn/
├── README.md             # This comprehensive guide to the project
├── main.py               # The main Python script for model training and prediction
├── sample_submission.csv # Example submission file (placeholder/format reference)
└── test.csv              # Test dataset used for generating predictions (actual data file)
```

---

## 📚 API Reference

This project is a script-based machine learning solution for a specific problem and does not expose a public API for programmatic interaction. Its primary interface is through executing the `main.py` script.

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please feel free to:

1.  **Fork** the repository. 🍴
2.  **Create** a new branch (`git checkout -b feature/AmazingFeature`).
3.  **Commit** your changes (`git commit -m 'Add some AmazingFeature'`).
4.  **Push** to the branch (`git push origin feature/AmazingFeature`).
5.  **Open** a Pull Request. 💬

---

## 📜 License

This project currently does not have an explicit license. Please contact the author for licensing information or consider contributing a suitable open-source license. ⚠️

---

## 🔗 Important Links

*   **GitHub Repository**: [https://github.com/VanshAgg1/Customer-Churn](https://github.com/VanshAgg1/Customer-Churn)
*   **Kaggle Challenge**: The dataset used in this project is typically sourced from a Kaggle competition related to bank customer churn. You can search Kaggle for "Bank Customer Churn Prediction" to find similar datasets. 🌐

---

## 👋 Footer

Repository: [Customer-Churn](https://github.com/VanshAgg1/Customer-Churn)

Developed by Vansh Aggarwal

Connect with the author: [VanshAgg1 on GitHub](https://github.com/VanshAgg1)

⭐️ Feel free to star this repository if you find it useful!

Fork the repository to start your own churn prediction journey! 🚀

Report issues or suggest improvements in the [Issues section](https://github.com/VanshAgg1/Customer-Churn/issues). 🐛



---
**<p align="center">Generated by [ReadmeCodeGen](https://www.readmecodegen.com/)</p>**