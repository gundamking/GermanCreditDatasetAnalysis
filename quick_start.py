#!/usr/bin/env python3
"""
Quick Start Script for German Credit Card Risk Analysis

This script provides a quick demonstration of the analysis capabilities.
Run this script to see the basic functionality of the project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def quick_demo():
    """Run a quick demonstration of the analysis."""
    print("🚀 German Credit Card Risk Analysis - Quick Demo")
    print("=" * 50)
    
    # Load data
    print("\n📊 Loading dataset...")
    try:
        df = pd.read_csv("german.data", delim_whitespace=True, header=None)
        columns = [
            'account_bal', 'duration', 'payment_status', 'purpose',
            'credit_amount', 'savings_bond_value', 'employed_since',
            'installment_rate', 'sex_marital', 'guarantor', 'residence_since',
            'most_valuable_asset', 'age', 'concurrent_credits', 'type_of_housing',
            'number_of_existcr', 'job', 'number_of_dependents', 'telephone',
            'foreign', 'target'
        ]
        df.columns = columns
        print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    except FileNotFoundError:
        print("❌ Error: german.data file not found!")
        print("Please ensure the dataset file is in the current directory.")
        return
    
    # Basic preprocessing
    print("\n🔧 Preprocessing data...")
    
    # Simple encoding for demo
    df['target'] = df['target'] - 1  # Convert to 0/1
    df['credit_amount'] = np.log(df['credit_amount'])
    
    # Select numerical features for quick demo
    numerical_features = ['duration', 'credit_amount', 'age', 'installment_rate']
    X = df[numerical_features]
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Training set: {X_train.shape}")
    print(f"✅ Test set: {X_test.shape}")
    
    # Train model
    print("\n🤖 Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Model accuracy: {accuracy:.3f}")
    
    # Feature importance
    print("\n📈 Feature Importance:")
    feature_importance = pd.DataFrame({
        'feature': numerical_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Basic visualization
    print("\n📊 Creating basic visualizations...")
    
    # Target distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    target_counts = df['target'].value_counts()
    plt.pie(target_counts.values, labels=['Good Credit', 'Bad Credit'], 
            autopct='%1.1f%%', startangle=90)
    plt.title('Credit Risk Distribution')
    
    # Credit amount distribution
    plt.subplot(1, 3, 2)
    plt.hist(df['credit_amount'], bins=30, alpha=0.7)
    plt.title('Credit Amount Distribution')
    plt.xlabel('Log Credit Amount')
    
    # Age distribution
    plt.subplot(1, 3, 3)
    plt.hist(df['age'], bins=30, alpha=0.7)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    
    plt.tight_layout()
    plt.savefig('images/quick_demo_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Visualizations saved to 'images/quick_demo_plots.png'")
    
    # Summary statistics
    print("\n📋 Dataset Summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Good credit: {sum(df['target'] == 0)} ({sum(df['target'] == 0)/len(df)*100:.1f}%)")
    print(f"  Bad credit: {sum(df['target'] == 1)} ({sum(df['target'] == 1)/len(df)*100:.1f}%)")
    print(f"  Average age: {df['age'].mean():.1f} years")
    print(f"  Average credit amount: {np.exp(df['credit_amount']).mean():.0f} DM")
    
    print("\n🎉 Quick demo completed!")
    print("\nTo run the full analysis:")
    print("  python analysis.py")
    print("\nTo explore the Jupyter notebook:")
    print("  jupyter notebook German-Credit-Card-Input-Data-Only.ipynb")


if __name__ == "__main__":
    quick_demo() 