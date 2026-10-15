import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Customer Churn Prediction for Subscription Services\n",
                "\n",
                "This notebook implements an end-to-end machine learning workflow for customer churn prediction using the Telco Customer Churn dataset. It follows a structured process from data loading to model comparison, feature importance, and business insights."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Import Libraries\n",
                "\n",
                "We import the libraries required for data handling, visualization, model training, evaluation, and result saving."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "import warnings\n",
                "from pathlib import Path\n",
                "\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score\n",
                "from sklearn.compose import ColumnTransformer\n",
                "from sklearn.pipeline import Pipeline\n",
                "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
                "from sklearn.impute import SimpleImputer\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.tree import DecisionTreeClassifier\n",
                "from sklearn.ensemble import RandomForestClassifier\n",
                "from xgboost import XGBClassifier\n",
                "from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, matthews_corrcoef\n",
                "from imblearn.over_sampling import SMOTE\n",
                "from sklearn.inspection import permutation_importance\n",
                "import joblib\n",
                "\n",
                "try:\n",
                "    import shap\n",
                "    shap_available = True\n",
                "except ImportError:\n",
                "    shap_available = False\n",
                "\n",
                "print('Libraries imported successfully.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Create Project Folders\n",
                "\n",
                "We create the required folders for dataset storage, notebook files, trained models, results, and figures."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "cwd = Path.cwd().resolve()\n",
                "if cwd.name == 'notebook' and (cwd.parent / 'Customer-Churn-Project').exists():\n",
                "    project_root = cwd.parent / 'Customer-Churn-Project'\n",
                "elif cwd.name == 'Customer-Churn-Project':\n",
                "    project_root = cwd\n",
                "elif (cwd / 'Customer-Churn-Project').exists():\n",
                "    project_root = cwd / 'Customer-Churn-Project'\n",
                "else:\n",
                "    project_root = cwd / 'Customer-Churn-Project'\n",
                "project_root = project_root.resolve()\n",
                "folders = ['data', 'notebook', 'models', 'results', 'figures']\n",
                "for folder in folders:\n",
                "    path = project_root / folder\n",
                "    path.mkdir(parents=True, exist_ok=True)\n",
                "    print(f'Created or verified: {path}')\n",
                "\n",
                "data_path = project_root / 'data' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'\n",
                "print('Dataset path set to:', data_path)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Load Dataset\n",
                "\n",
                "We load the dataset from the data folder and inspect the first rows, shape, columns, and data types."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "df = pd.read_csv(data_path)\n",
                "print('Dataset loaded successfully.')\n",
                "print('Shape:', df.shape)\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "print('Column names:')\n",
                "print(df.columns.tolist())\n",
                "print('\\nData types:')\n",
                "print(df.dtypes)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "The dataset contains customer subscription information, account details, and whether the customer churned. The target variable is `Churn`, which indicates customer cancellation."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Initial Data Understanding\n",
                "\n",
                "We inspect data summary statistics, missing values, duplicates, unique counts, and identify numerical and categorical columns."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "print('Data information:')\n",
                "df.info()\n",
                "\n",
                "print('Summary statistics for numerical columns:')\n",
                "df.describe().T"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "missing_values = df.isnull().sum()\n",
                "print('Missing values per column:')\n",
                "print(missing_values[missing_values > 0])\n",
                "\n",
                "duplicates = df.duplicated().sum()\n",
                "print('Duplicate rows:', duplicates)\n",
                "\n",
                "print('Target variable distribution:')\n",
                "print(df['Churn'].value_counts())\n",
                "print('\\nTarget distribution percentage:')\n",
                "print(df['Churn'].value_counts(normalize=True) * 100)"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "categorical_cols = df.select_dtypes(include=['object']).columns.tolist()\n",
                "numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()\n",
                "print('Categorical columns:', categorical_cols)\n",
                "print('Numerical columns:', numerical_cols)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "The dataset has a mixture of numerical and categorical features. We will clean the data and convert types where needed before analysis and modeling."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Data Cleaning\n",
                "\n",
                "We remove duplicates, convert `TotalCharges` to numeric, handle missing values, and ensure correct feature types."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "df_clean = df.copy()\n",
                "df_clean = df_clean.drop_duplicates().reset_index(drop=True)\n",
                "print('Rows after removing duplicates:', df_clean.shape[0])\n",
                "\n",
                "df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')\n",
                "missing_total = df_clean['TotalCharges'].isnull().sum()\n",
                "print('Missing TotalCharges after conversion:', missing_total)\n",
                "\n",
                "df_clean = df_clean.dropna(subset=['TotalCharges']).reset_index(drop=True)\n",
                "print('Rows after dropping missing TotalCharges:', df_clean.shape[0])\n",
                "\n",
                "df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].astype('int64')\n",
                "df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})\n",
                "print('Data types after cleaning:')\n",
                "print(df_clean.dtypes)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "The `TotalCharges` column was stored as text and contained a few missing values. We converted it to numeric and removed rows that could not be converted. The target variable was encoded to binary format for modeling."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Exploratory Data Analysis (EDA)\n",
                "\n",
                "We explore the target distribution, numerical feature distributions, categorical relationships, and correlations. Figures are saved to the `figures/` folder."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "figure_path = project_root / 'figures'\n",
                "figure_path.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "plt.figure(figsize=(6, 4))\n",
                "sns.countplot(data=df_clean, x='Churn', palette='Set2')\n",
                "plt.title('Target variable: Churn count')\n",
                "plt.xticks([0, 1], ['No Churn', 'Churn'])\n",
                "plt.ylabel('Count')\n",
                "plt.xlabel('Churn')\n",
                "plt.tight_layout()\n",
                "plt.savefig(figure_path / 'target_churn_distribution.png', dpi=150)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "This plot shows the imbalance between customers who stayed and customers who churned. Understanding this imbalance is important for model training and evaluation."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "num_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']\n",
                "for col in num_columns:\n",
                "    plt.figure(figsize=(10, 4))\n",
                "    sns.histplot(df_clean[col], kde=True, color='steelblue')\n",
                "    plt.title(f'Distribution of {col}')\n",
                "    plt.xlabel(col)\n",
                "    plt.tight_layout()\n",
                "    plt.savefig(figure_path / f'hist_{col}.png', dpi=150)\n",
                "    plt.show()\n",
                "    \n",
                "    plt.figure(figsize=(10, 4))\n",
                "    sns.boxplot(x=df_clean[col], color='lightgreen')\n",
                "    plt.title(f'Boxplot of {col}')\n",
                "    plt.tight_layout()\n",
                "    plt.savefig(figure_path / f'box_{col}.png', dpi=150)\n",
                "    plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "These plots show the numerical distributions and identify any potential outliers in tenure, monthly charges, and total charges."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "categorical_analysis = ['Contract', 'InternetService', 'PaymentMethod', 'gender', 'SeniorCitizen', 'OnlineSecurity', 'TechSupport']\n",
                "for col in categorical_analysis:\n",
                "    plt.figure(figsize=(10, 5))\n",
                "    sns.countplot(data=df_clean, x=col, hue='Churn', palette='Set1')\n",
                "    plt.title(f'Churn by {col}')\n",
                "    plt.legend(title='Churn', labels=['No', 'Yes'])\n",
                "    plt.xticks(rotation=45)\n",
                "    plt.tight_layout()\n",
                "    plt.savefig(figure_path / f'churn_by_{col}.png', dpi=150)\n",
                "    plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "These categorical plots show how churn relates to contract type, internet service, payment method, gender, senior citizen status, online security, and tech support. This helps identify important business drivers."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "plt.figure(figsize=(12, 10))\n",
                "corr = df_clean[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']].corr()\n",
                "sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')\n",
                "plt.title('Correlation heatmap for numerical features')\n",
                "plt.tight_layout()\n",
                "plt.savefig(figure_path / 'correlation_heatmap.png', dpi=150)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "The correlation heatmap shows relationships between numerical features and the target. Strong correlations help identify which numerical factors are most associated with churn."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Feature Engineering\n",
                "\n",
                "We create new features such as tenure groups and monthly charge categories to help the model capture non-linear patterns."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "df_features = df_clean.copy()\n",
                "df_features['tenure_group'] = pd.cut(df_features['tenure'], bins=[0, 12, 24, 48, 72], labels=['0-12', '13-24', '25-48', '49-72'])\n",
                "df_features['MonthlyCharges_group'] = pd.cut(df_features['MonthlyCharges'], bins=[0, 35, 70, 120], labels=['Low', 'Medium', 'High'])\n",
                "df_features['TotalCharges_group'] = pd.qcut(df_features['TotalCharges'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])\n",
                "print('Feature engineering completed.')\n",
                "df_features[['tenure_group', 'MonthlyCharges_group', 'TotalCharges_group']].head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Grouping tenure and charges can help the model detect patterns that depend on customer lifecycle stages and spending categories. These features can also support business discussions in the dissertation."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Data Preprocessing\n",
                "\n",
                "We prepare data for modeling by encoding categorical variables, scaling numerical columns, and assembling a pipeline to avoid leakage."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "X = df_features.drop(columns=['customerID', 'Churn'])\n",
                "y = df_features['Churn']\n",
                "\n",
                "numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']\n",
                "categorical_features = [col for col in X.columns if X[col].dtype == 'object' or col == 'SeniorCitizen']\n",
                "\n",
                "numeric_transformer = Pipeline(steps=[\n",
                "    ('imputer', SimpleImputer(strategy='median')),\n",
                "    ('scaler', StandardScaler())\n",
                "])\n",
                "categorical_transformer = Pipeline(steps=[\n",
                "    ('imputer', SimpleImputer(strategy='most_frequent')),\n",
                "    ('onehot', OneHotEncoder(handle_unknown='ignore'))\n",
                "])\n",
                "\n",
                "preprocessor = ColumnTransformer(transformers=[\n",
                "    ('num', numeric_transformer, numeric_features),\n",
                "    ('cat', categorical_transformer, categorical_features),\n",
                "])\n",
                "\n",
                "print('Preprocessing pipeline defined.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "OneHotEncoder converts categorical values to numeric format, and StandardScaler scales numeric features. The pipeline keeps preprocessing separate from model fitting."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Train-Test Split\n",
                "\n",
                "We split data into training and test sets while preserving the target distribution using stratification."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "X_train, X_test, y_train, y_test = train_test_split(\n",
                "    X, y, test_size=0.2, random_state=42, stratify=y\n",
                ")\n",
                "\n",
                "print('Training set shape:', X_train.shape)\n",
                "print('Test set shape:', X_test.shape)\n",
                "print('Train churn distribution:')\n",
                "print(y_train.value_counts(normalize=True))\n",
                "print('Test churn distribution:')\n",
                "print(y_test.value_counts(normalize=True))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Stratification ensures the training and test sets preserve the same churn rate, which is important for reliable evaluation in imbalanced data."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 10. Handle Class Imbalance\n",
                "\n",
                "We use SMOTE to synthesize new minority class samples only on the training data."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "print('Class distribution before SMOTE:')\n",
                "print(y_train.value_counts())\n",
                "\n",
                "X_train_processed = preprocessor.fit_transform(X_train)\n",
                "X_test_processed = preprocessor.transform(X_test)\n",
                "\n",
                "smote = SMOTE(random_state=42)\n",
                "X_train_res, y_train_res = smote.fit_resample(X_train_processed, y_train)\n",
                "\n",
                "print('Class distribution after SMOTE:')\n",
                "print(pd.Series(y_train_res).value_counts())\n",
                "\n",
                "plt.figure(figsize=(6, 4))\n",
                "sns.countplot(x=y_train, palette='Set2')\n",
                "plt.title('Training class distribution before SMOTE')\n",
                "plt.xlabel('Churn')\n",
                "plt.ylabel('Count')\n",
                "plt.tight_layout()\n",
                "plt.savefig(figure_path / 'smote_before.png', dpi=150)\n",
                "plt.show()\n",
                "\n",
                "plt.figure(figsize=(6, 4))\n",
                "sns.countplot(x=y_train_res, palette='Set2')\n",
                "plt.title('Training class distribution after SMOTE')\n",
                "plt.xlabel('Churn')\n",
                "plt.ylabel('Count')\n",
                "plt.tight_layout()\n",
                "plt.savefig(figure_path / 'smote_after.png', dpi=150)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "SMOTE creates synthetic examples of churn cases so the model can learn patterns from a balanced training set. This is important because churn is the minority class."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 11. Model Training\n",
                "\n",
                "We train Logistic Regression, Decision Tree, Random Forest, and XGBoost models using the balanced training data."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "models = {\n",
                "    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),\n",
                "    'Decision Tree': DecisionTreeClassifier(random_state=42),\n",
                "    'Random Forest': RandomForestClassifier(random_state=42),\n",
                "    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)\n",
                "}\n",
                "trained_models = {}\n",
                "predictions = {}\n",
                "probabilities = {}\n",
                "\n",
                "for name, model in models.items():\n",
                "    model.fit(X_train_res, y_train_res)\n",
                "    trained_models[name] = model\n",
                "    y_pred = model.predict(X_test_processed)\n",
                "    y_proba = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test_processed)\n",
                "    predictions[name] = y_pred\n",
                "    probabilities[name] = y_proba\n",
                "    model_file = project_root / 'models' / f\"{name.replace(' ', '_').lower()}_model.pkl\"\n",
                "    joblib.dump(model, model_file)\n",
                "    print(f'{name} trained and saved.')\n",
                "\n",
                "print('All base models trained.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "All models are trained on the SMOTE-balanced training set. We saved each trained model for reproducibility and future evaluation."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 12. Hyperparameter Tuning\n",
                "\n",
                "We tune Decision Tree, Random Forest, and XGBoost using GridSearchCV with recall scoring. Recall is important because we want to identify churn cases accurately."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "param_grids = {\n",
                "    'Decision Tree': {\n",
                "        'max_depth': [4, 6, 8, 10],\n",
                "        'min_samples_split': [10, 20, 30],\n",
                "    },\n",
                "    'Random Forest': {\n",
                "        'n_estimators': [100, 200],\n",
                "        'max_depth': [6, 8, 10],\n",
                "        'min_samples_split': [10, 20],\n",
                "    },\n",
                "    'XGBoost': {\n",
                "        'n_estimators': [100, 150],\n",
                "        'max_depth': [3, 5, 7],\n",
                "        'learning_rate': [0.05, 0.1],\n",
                "    },\n",
                "}\n",
                "best_models = {}\n",
                "\n",
                "for name in ['Decision Tree', 'Random Forest', 'XGBoost']:\n",
                "    print(f'Tuning {name}...')\n",
                "    grid = GridSearchCV(models[name], param_grids[name], cv=5, scoring='recall', n_jobs=-1)\n",
                "    grid.fit(X_train_res, y_train_res)\n",
                "    best_models[name] = grid.best_estimator_\n",
                "    best_file = project_root / 'models' / f\"{name.replace(' ', '_').lower()}_best.pkl\"\n",
                "    joblib.dump(grid.best_estimator_, best_file)\n",
                "    print(f'{name} best params:', grid.best_params_)\n",
                "    print(f'{name} best recall:', grid.best_score_)\n",
                "\n",
                "print('Hyperparameter tuning completed.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Tuning models with recall helps prioritize detection of churn cases, which is critical for customer retention strategies."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 13. Cross Validation\n",
                "\n",
                "We evaluate each model using Stratified K-Fold cross-validation and report average performance metrics."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
                "cv_results = []\n",
                "\n",
                "def evaluate_cv(model, X, y, cv):\n",
                "    scoring = {\n",
                "        'accuracy': 'accuracy',\n",
                "        'precision': 'precision',\n",
                "        'recall': 'recall',\n",
                "        'f1': 'f1',\n",
                "    }\n",
                "    results = {}\n",
                "    for score_name, score_method in scoring.items():\n",
                "        results[score_name] = cross_val_score(model, X, y, cv=cv, scoring=score_method, n_jobs=-1).mean()\n",
                "    y_pred = model.predict(X_test_processed)\n",
                "    y_proba = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test_processed)\n",
                "    results['roc_auc'] = roc_auc_score(y_test, y_proba)\n",
                "    return results\n",
                "\n",
                "for name, model in trained_models.items():\n",
                "    print(f'Evaluating with cross validation: {name}')\n",
                "    scores = evaluate_cv(model, X_train_res, y_train_res, cv)\n",
                "    cv_results.append({'Model': name, **scores})\n",
                "\n",
                "for name, model in best_models.items():\n",
                "    print(f'Evaluating tuned model: {name}')\n",
                "    scores = evaluate_cv(model, X_train_res, y_train_res, cv)\n",
                "    cv_results.append({'Model': f'{name} (Tuned)', **scores})\n",
                "\n",
                "cv_df = pd.DataFrame(cv_results)\n",
                "cv_df"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Cross validation helps estimate how the models will generalize to new data. We report mean metrics across the five folds."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 14. Model Evaluation\n",
                "\n",
                "We evaluate each model on the test set with metrics, confusion matrices, ROC curves, and precision-recall curves."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "evaluation_records = []\n",
                "\n",
                "for name, model in {**trained_models, **best_models}.items():\n",
                "    y_pred = model.predict(X_test_processed)\n",
                "    y_proba = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test_processed)\n",
                "    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)\n",
                "    cm = confusion_matrix(y_test, y_pred)\n",
                "    auc = roc_auc_score(y_test, y_proba)\n",
                "    evaluation_records.append({\n",
                "        'Model': name,\n",
                "        'Accuracy': accuracy_score(y_test, y_pred),\n",
                "        'Precision': precision_score(y_test, y_pred),\n",
                "        'Recall': recall_score(y_test, y_pred),\n",
                "        'F1-score': f1_score(y_test, y_pred),\n",
                "        'ROC-AUC': auc,\n",
                "        'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred),\n",
                "        'MCC': matthews_corrcoef(y_test, y_pred),\n",
                "    })\n",
                "    plt.figure(figsize=(5, 4))\n",
                "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\n",
                "    plt.title(f'Confusion Matrix: {name}')\n",
                "    plt.xlabel('Predicted')\n",
                "    plt.ylabel('Actual')\n",
                "    plt.tight_layout()\n",
                "    plt.savefig(figure_path / f'confusion_matrix_{name.replace(' ', '_').lower()}.png', dpi=150)\n",
                "    plt.show()\n",
                "    \n",
                "    fpr, tpr, _ = roc_curve(y_test, y_proba)\n",
                "    plt.figure(figsize=(6, 4))\n",
                "    plt.plot(fpr, tpr, label=f'ROC AUC = {auc:.3f}')\n",
                "    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')\n",
                "    plt.title(f'ROC Curve: {name}')\n",
                "    plt.xlabel('False Positive Rate')\n",
                "    plt.ylabel('True Positive Rate')\n",
                "    plt.legend()\n",
                "    plt.tight_layout()\n",
                "    plt.savefig(figure_path / f'roc_curve_{name.replace(' ', '_').lower()}.png', dpi=150)\n",
                "    plt.show()\n",
                "    \n",
                "    precision, recall, _ = precision_recall_curve(y_test, y_proba)\n",
                "    plt.figure(figsize=(6, 4))\n",
                "    plt.plot(recall, precision)\n",
                "    plt.title(f'Precision-Recall Curve: {name}')\n",
                "    plt.xlabel('Recall')\n",
                "    plt.ylabel('Precision')\n",
                "    plt.tight_layout()\n",
                "    plt.savefig(figure_path / f'precision_recall_{name.replace(' ', '_').lower()}.png', dpi=150)\n",
                "    plt.show()\n",
                "    \n",
                "    report_df = pd.DataFrame(report).T\n",
                "    report_df.to_csv(project_root / 'results' / f'classification_report_{name.replace(' ', '_').lower()}.csv')\n",
                "\n",
                "evaluation_df = pd.DataFrame(evaluation_records)\n",
                "evaluation_df.to_csv(project_root / 'results' / 'model_evaluation_summary.csv', index=False)\n",
                "evaluation_df"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "These evaluation results compare how each model performs on the test set and save the metrics and figures for the dissertation."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 15. Model Comparison\n",
                "\n",
                "We create a comparison table and charts to rank models by recall and other metrics."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "comparison_df = evaluation_df.sort_values(by='Recall', ascending=False).reset_index(drop=True)\n",
                "comparison_df.to_csv(project_root / 'results' / 'model_comparison_sorted.csv', index=False)\n",
                "comparison_df\n",
                "\n",
                "plt.figure(figsize=(10, 6))\n",
                "sns.barplot(data=comparison_df, x='Recall', y='Model', palette='viridis')\n",
                "plt.title('Model comparison by Recall')\n",
                "plt.tight_layout()\n",
                "plt.savefig(figure_path / 'model_comparison_recall.png', dpi=150)\n",
                "plt.show()\n",
                "\n",
                "metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']\n",
                "plt.figure(figsize=(12, 6))\n",
                "for metric in metrics_to_plot:\n",
                "    plt.plot(comparison_df['Model'], comparison_df[metric], marker='o', label=metric)\n",
                "plt.title('Model performance metrics')\n",
                "plt.xlabel('Model')\n",
                "plt.xticks(rotation=45)\n",
                "plt.legend()\n",
                "plt.tight_layout()\n",
                "plt.savefig(figure_path / 'model_comparison_metrics.png', dpi=150)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "This comparison highlights the best model for predicting churn and provides a visual summary of key metrics."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 16. Feature Importance Analysis\n",
                "\n",
                "We analyze feature importance for the tree-based models and display the top features that influence churn."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "def plot_feature_importance(model, model_name, feature_names, top_n=20):\n",
                "    importances = model.feature_importances_\n",
                "    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})\n",
                "    importance_df = importance_df.sort_values(by='importance', ascending=False).head(top_n)\n",
                "    plt.figure(figsize=(10, 6))\n",
                "    sns.barplot(data=importance_df, x='importance', y='feature', palette='rocket')\n",
                "    plt.title(f'Top {top_n} features for {model_name}')\n",
                "    plt.tight_layout()\n",
                "    plt.savefig(figure_path / f\"feature_importance_{model_name.replace(' ', '_').lower()}.png\", dpi=150)\n",
                "    plt.show()\n",
                "    return importance_df\n",
                "\n",
                "feature_names = numeric_features + preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features).tolist()\n",
                "importance_results = {}\n",
                "for model_name in ['Decision Tree', 'Random Forest', 'XGBoost']:\n",
                "    model = best_models.get(model_name, trained_models[model_name])\n",
                "    importance_df = plot_feature_importance(model, model_name, feature_names)\n",
                "    importance_df.to_csv(project_root / 'results' / f\"feature_importance_{model_name.replace(' ', '_').lower()}.csv\", index=False)\n",
                "    importance_results[model_name] = importance_df\n",
                "\n",
                "importance_results['Random Forest'].head(20)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Feature importance for tree-based models helps identify the customer attributes that most strongly influence churn. These features are valuable for business strategy."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 17. Permutation Importance\n",
                "\n",
                "We compute permutation importance for the best XGBoost model to measure how each feature affects model performance."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "best_xgb = best_models['XGBoost']\n",
                "perm_result = permutation_importance(best_xgb, X_test_processed, y_test, n_repeats=10, random_state=42, n_jobs=-1)\n",
                "perm_df = pd.DataFrame({'feature': feature_names, 'importance': perm_result.importances_mean})\n",
                "perm_df = perm_df.sort_values(by='importance', ascending=False).head(20)\n",
                "plt.figure(figsize=(10, 6))\n",
                "sns.barplot(data=perm_df, x='importance', y='feature', palette='mako')\n",
                "plt.title('Top 20 Permutation Importances for XGBoost')\n",
                "plt.tight_layout()\n",
                "plt.savefig(figure_path / 'permutation_importance_xgboost.png', dpi=150)\n",
                "plt.show()\n",
                "perm_df.to_csv(project_root / 'results' / 'permutation_importance_xgboost.csv', index=False)\n",
                "perm_df"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Permutation importance shows how much the model relies on each feature for prediction accuracy. This helps validate the importance ranking from tree-based models."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 18. SHAP Explainability\n",
                "\n",
                "We attempt SHAP explainability for the XGBoost model to produce global and local interpretations."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "if shap_available:\n",
                "    explainer = shap.TreeExplainer(best_xgb)\n",
                "    shap_values = explainer.shap_values(X_test_processed)\n",
                "    plt.figure(figsize=(10, 7))\n",
                "    shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names, show=False)\n",
                "    plt.tight_layout()\n",
                "    plt.savefig(figure_path / 'shap_summary_xgboost.png', dpi=150)\n",
                "    plt.close()\n",
                "    print('SHAP summary plot saved.')\n",
                "    \n",
                "    plt.figure(figsize=(10, 7))\n",
                "    shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names, plot_type='bar', show=False)\n",
                "    plt.tight_layout()\n",
                "    plt.savefig(figure_path / 'shap_bar_xgboost.png', dpi=150)\n",
                "    plt.close()\n",
                "    print('SHAP bar plot saved.')\n",
                "    \n",
                "    sample_index = 0\n",
                "    shap.force_plot(explainer.expected_value, shap_values[sample_index], X_test_processed[sample_index], feature_names=feature_names, matplotlib=True, show=False)\n",
                "    plt.tight_layout()\n",
                "    plt.savefig(figure_path / 'shap_force_plot_sample.png', dpi=150)\n",
                "    plt.close()\n",
                "    print('SHAP force plot saved for one example.')\n",
                "else:\n",
                "    print('SHAP is not available. Skipping SHAP plots.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "SHAP explains model predictions at the feature level. If SHAP is unavailable, the notebook continues without it."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 19. Business Insights\n",
                "\n",
                "We summarize customer churn behavior, key predictors, and retention actions."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "- Customers with short tenure, high monthly charges, and month-to-month contracts are more likely to churn.\n",
                "- Churn is often higher for customers without tech support or online security services.\n",
                "- Business retention strategies should focus on flexible contracts, service bundles, and improved support for at-risk customers.\n",
                "- The model can help customer success teams target offers to customers who are likely to cancel."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 20. Final Conclusion\n",
                "\n",
                "The best performing model is selected based on recall and overall balance of metrics. Important evaluation outcomes, churn drivers, limitations, and future improvements are summarized."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "- The best model should be the one with the highest recall and strong ROC-AUC.\n",
                "- Important churn drivers include contract type, tenure, monthly charges, and service support features.\n",
                "- The business impact is improved customer retention and better targeting of retention offers.\n",
                "- Limitations include dataset bias, feature availability, and imbalanced churn rates.\n",
                "- Future work can include additional customer behavior data, time-series modeling, and personalized retention campaigns."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

output_path = Path('customer_churn_analysis.ipynb')
output_path.write_text(json.dumps(notebook, indent=1), encoding='utf-8')
print('Notebook rewritten at', output_path.resolve())
