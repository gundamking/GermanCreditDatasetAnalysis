#!/usr/bin/env python3
"""
German Credit Card Risk Analysis

This script performs comprehensive analysis and machine learning modeling
on the German Credit Card dataset to predict credit risk.

Author: gundamking
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, accuracy_score
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class GermanCreditAnalysis:
    """
    A comprehensive class for analyzing the German Credit Card dataset
    and building machine learning models for credit risk prediction.
    """
    
    def __init__(self, data_path='german.data'):
        """
        Initialize the analysis with the dataset path.
        
        Args:
            data_path (str): Path to the German credit dataset
        """
        self.data_path = data_path
        self.credit_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
        # Define column names based on dataset documentation
        self.columns = [
            'account_bal', 'duration', 'payment_status', 'purpose',
            'credit_amount', 'savings_bond_value', 'employed_since',
            'installment_rate', 'sex_marital', 'guarantor', 'residence_since',
            'most_valuable_asset', 'age', 'concurrent_credits', 'type_of_housing',
            'number_of_existcr', 'job', 'number_of_dependents', 'telephone',
            'foreign', 'target'
        ]
        
        # Define value mappings for categorical variables
        self.value_mappings = {
            'account_bal': {
                'A11': 'neg_bal', 'A12': 'positive_bal', 
                'A13': 'positive_bal', 'A14': 'no_acc'
            },
            'payment_status': {
                'A30': 'no_credits', 'A31': 'all_paid', 
                'A32': 'existing_paid', 'A33': 'delay_past', 'A34': 'critical'
            },
            'purpose': {
                'A40': 'car_new', 'A41': 'car_used', 'A42': 'furniture',
                'A43': 'radio_tv', 'A44': 'appliances', 'A45': 'repairs',
                'A46': 'education', 'A47': 'vacation', 'A48': 'retraining',
                'A49': 'business', 'A410': 'others'
            },
            'savings_bond_value': {
                'A61': 'less_100', 'A62': '100_to_500', 'A63': '500_to_1000',
                'A64': 'more_1000', 'A65': 'unknown'
            },
            'employed_since': {
                'A71': 'unemployed', 'A72': 'less_1_year', 'A73': '1_to_4_years',
                'A74': '4_to_7_years', 'A75': 'more_7_years'
            },
            'sex_marital': {
                'A91': 'male_divorced', 'A92': 'female_divorced', 'A93': 'male_single',
                'A94': 'male_married', 'A95': 'female_single'
            },
            'guarantor': {
                'A101': 'none', 'A102': 'co_applicant', 'A103': 'guarantor'
            },
            'most_valuable_asset': {
                'A121': 'real_estate', 'A122': 'life_insurance', 
                'A123': 'car_other', 'A124': 'unknown'
            },
            'type_of_housing': {
                'A141': 'bank', 'A142': 'stores', 'A143': 'none'
            },
            'housing': {
                'A151': 'rent', 'A152': 'own', 'A153': 'free'
            },
            'job': {
                'A171': 'unskilled_non_resident', 'A172': 'unskilled_resident',
                'A173': 'skilled', 'A174': 'management'
            },
            'telephone': {
                'A191': 'none', 'A192': 'yes'
            },
            'foreign': {
                'A201': 'yes', 'A202': 'no'
            }
        }
    
    def load_data(self):
        """Load and preprocess the German credit dataset."""
        print("Loading German Credit Dataset...")
        
        # Load the data
        self.credit_df = pd.read_csv(self.data_path, delim_whitespace=True, header=None)
        self.credit_df.columns = self.columns
        
        print(f"Dataset loaded: {self.credit_df.shape[0]} samples, {self.credit_df.shape[1]} features")
        return self
    
    def preprocess_data(self):
        """Preprocess the data including encoding and feature engineering."""
        print("Preprocessing data...")
        
        # Apply value mappings
        for column, mapping in self.value_mappings.items():
            if column in self.credit_df.columns:
                self.credit_df[column] = self.credit_df[column].map(mapping)
        
        # Log transform credit amount for better distribution
        self.credit_df['credit_amount'] = np.log(self.credit_df['credit_amount'])
        
        # Encode categorical variables
        self._encode_categorical_variables()
        
        # Create polynomial features for important numerical variables
        self._create_polynomial_features()
        
        print("Data preprocessing completed.")
        return self
    
    def _encode_categorical_variables(self):
        """Encode categorical variables using label encoding and one-hot encoding."""
        # Label encode binary categorical variables
        le = LabelEncoder()
        binary_columns = []
        
        for col in self.credit_df.select_dtypes('object').columns:
            if len(self.credit_df[col].unique()) <= 2:
                self.credit_df[col] = le.fit_transform(self.credit_df[col])
                binary_columns.append(col)
        
        # One-hot encode remaining categorical variables
        categorical_columns = self.credit_df.select_dtypes('object').columns
        if len(categorical_columns) > 0:
            self.credit_df = pd.get_dummies(self.credit_df, columns=categorical_columns, drop_first=True)
    
    def _create_polynomial_features(self):
        """Create polynomial features for important numerical variables."""
        # Select important numerical features for polynomial features
        important_features = ['duration', 'credit_amount', 'age']
        available_features = [f for f in important_features if f in self.credit_df.columns]
        
        if len(available_features) >= 2:
            poly_features = self.credit_df[available_features]
            poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
            poly_features_transformed = poly_transformer.fit_transform(poly_features)
            
            # Create DataFrame with polynomial features
            poly_df = pd.DataFrame(
                poly_features_transformed,
                columns=poly_transformer.get_feature_names_out(available_features)
            )
            
            # Remove duplicate columns
            for col in poly_df.columns:
                if col in self.credit_df.columns:
                    poly_df = poly_df.drop(columns=[col])
            
            # Add polynomial features to main dataset
            self.credit_df = pd.concat([self.credit_df, poly_df], axis=1)
    
    def explore_data(self):
        """Perform exploratory data analysis."""
        print("Performing Exploratory Data Analysis...")
        
        # Basic statistics
        print("\nDataset Overview:")
        print(f"Shape: {self.credit_df.shape}")
        print(f"Target distribution:\n{self.credit_df['target'].value_counts()}")
        
        # Correlation analysis
        self._analyze_correlations()
        
        # Create visualizations
        self._create_visualizations()
        
        return self
    
    def _analyze_correlations(self):
        """Analyze correlations with target variable."""
        correlations = self.credit_df.corr()['target'].sort_values()
        
        print("\nTop 10 Positive Correlations:")
        print(correlations.tail(10))
        
        print("\nTop 10 Negative Correlations:")
        print(correlations.head(10))
    
    def _create_visualizations(self):
        """Create various visualizations for data exploration."""
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Target distribution
        plt.figure(figsize=(10, 6))
        target_counts = self.credit_df['target'].value_counts()
        plt.pie(target_counts.values, labels=['Good Credit', 'Bad Credit'], 
                autopct='%1.1f%%', startangle=90)
        plt.title('Credit Risk Distribution')
        plt.savefig('images/target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Correlation heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.credit_df.corr()
        sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('images/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Numerical features distribution
        numerical_features = ['duration', 'credit_amount', 'age']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, feature in enumerate(numerical_features):
            if feature in self.credit_df.columns:
                axes[i].hist(self.credit_df[feature], bins=30, alpha=0.7)
                axes[i].set_title(f'{feature.replace("_", " ").title()} Distribution')
                axes[i].set_xlabel(feature.replace("_", " ").title())
                axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('images/numerical_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def prepare_modeling_data(self):
        """Prepare data for machine learning modeling."""
        print("Preparing data for modeling...")
        
        # Separate features and target
        X = self.credit_df.drop('target', axis=1)
        y = self.credit_df['target']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        return self
    
    def build_models(self):
        """Build and train multiple machine learning models."""
        print("Building and training models...")
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # Train models
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
        
        return self
    
    def evaluate_models(self):
        """Evaluate all models and generate performance metrics."""
        print("Evaluating models...")
        
        for name, model in self.models.items():
            print(f"\n{name} Results:")
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else None
            
            print(f"Accuracy: {accuracy:.4f}")
            if auc:
                print(f"AUC: {auc:.4f}")
            
            # Classification report
            print("Classification Report:")
            print(classification_report(self.y_test, y_pred))
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
        
        return self
    
    def create_model_comparison_plots(self):
        """Create comparison plots for model performance."""
        print("Creating model comparison plots...")
        
        # ROC Curves
        plt.figure(figsize=(10, 8))
        for name, result in self.results.items():
            if result['probabilities'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, result['probabilities'])
                plt.plot(fpr, tpr, label=f"{name} (AUC = {result['auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig('images/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Model performance comparison
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        aucs = [self.results[name]['auc'] for name in model_names if self.results[name]['auc'] is not None]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        ax1.bar(model_names, accuracies, color='skyblue')
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        
        # AUC comparison
        if aucs:
            auc_model_names = [name for name in model_names if self.results[name]['auc'] is not None]
            ax2.bar(auc_model_names, aucs, color='lightcoral')
            ax2.set_title('Model AUC Comparison')
            ax2.set_ylabel('AUC')
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('images/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """Save model results and performance metrics."""
        print("Saving results...")
        
        # Save results to CSV
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[name]['accuracy'] for name in self.results.keys()],
            'AUC': [self.results[name]['auc'] for name in self.results.keys()]
        })
        
        results_df.to_csv('results/model_performance.csv', index=False)
        
        # Save detailed classification reports
        with open('results/classification_reports.txt', 'w') as f:
            for name, result in self.results.items():
                f.write(f"\n{'='*50}\n")
                f.write(f"{name} Classification Report\n")
                f.write(f"{'='*50}\n")
                f.write(classification_report(self.y_test, result['predictions']))
        
        print("Results saved to 'results/' directory.")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting German Credit Card Risk Analysis...")
        print("="*50)
        
        (self.load_data()
         .preprocess_data()
         .explore_data()
         .prepare_modeling_data()
         .build_models()
         .evaluate_models()
         .create_model_comparison_plots()
         .save_results())
        
        print("\nAnalysis completed successfully!")
        print("Check the 'images/' directory for visualizations")
        print("Check the 'results/' directory for performance metrics")


def main():
    """Main function to run the analysis."""
    # Initialize the analysis
    analysis = GermanCreditAnalysis()
    
    # Run complete analysis
    analysis.run_complete_analysis()


if __name__ == "__main__":
    main() 