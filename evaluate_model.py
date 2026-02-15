#!/usr/bin/env python3
"""
Model Evaluation and Cross-Validation
Advanced analysis of ML model performance
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class ModelEvaluator:
    """
    Advanced model evaluation and diagnostics
    """
    
    def __init__(self, project_dir='../'):
        self.project_dir = project_dir
        self.models_dir = os.path.join(project_dir, 'models')
        self.results_dir = os.path.join(project_dir, 'results')
        self.figures_dir = os.path.join(self.results_dir, 'figures')
        
        os.makedirs(self.figures_dir, exist_ok=True)
    
    def cross_validate_model(self, X, y, model=None, cv=5):
        """
        Perform k-fold cross-validation
        
        Args:
            X: Feature matrix
            y: Target values
            model: Sklearn model (if None, loads saved model)
            cv: Number of folds
            
        Returns:
            Dictionary with CV metrics
        """
        if model is None:
            model_path = os.path.join(self.models_dir, 'adrb2_rf_model.pkl')
            model = joblib.load(model_path)
        
        print(f"\nPerforming {cv}-fold cross-validation...")
        
        # Define scoring metrics
        scoring = {
            'r2': 'r2',
            'neg_mse': 'neg_mean_squared_error',
            'neg_mae': 'neg_mean_absolute_error'
        }
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
            verbose=1
        )
        
        # Calculate statistics
        metrics = {
            'cv_r2_mean': cv_results['test_r2'].mean(),
            'cv_r2_std': cv_results['test_r2'].std(),
            'cv_rmse_mean': np.sqrt(-cv_results['test_neg_mse'].mean()),
            'cv_rmse_std': np.sqrt(-cv_results['test_neg_mse']).std(),
            'cv_mae_mean': -cv_results['test_neg_mae'].mean(),
            'cv_mae_std': (-cv_results['test_neg_mae']).std(),
            'train_r2_mean': cv_results['train_r2'].mean(),
            'train_r2_std': cv_results['train_r2'].std()
        }
        
        print("\n=== Cross-Validation Results ===")
        print(f"R² Score: {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
        print(f"RMSE: {metrics['cv_rmse_mean']:.4f} ± {metrics['cv_rmse_std']:.4f}")
        print(f"MAE: {metrics['cv_mae_mean']:.4f} ± {metrics['cv_mae_std']:.4f}")
        
        # Check for overfitting
        overfit_margin = metrics['train_r2_mean'] - metrics['cv_r2_mean']
        print(f"\nOverfitting check:")
        print(f"Train-Test R² gap: {overfit_margin:.4f}")
        if overfit_margin > 0.15:
            print("⚠️  Warning: Possible overfitting detected")
        else:
            print("✅ Model generalization looks good")
        
        return metrics, cv_results
    
    def plot_learning_curve(self, X, y, model=None, save=True):
        """
        Generate learning curve to diagnose bias/variance
        
        Args:
            X: Feature matrix
            y: Target values
            model: Sklearn model
            save: Save figure
        """
        from sklearn.model_selection import learning_curve
        
        if model is None:
            model_path = os.path.join(self.models_dir, 'adrb2_rf_model.pkl')
            model = joblib.load(model_path)
        
        print("\nGenerating learning curve...")
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes_abs, train_scores, test_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        # Calculate mean and std
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        test_mean = test_scores.mean(axis=1)
        test_std = test_scores.std(axis=1)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(train_sizes_abs, train_mean, 'o-', color='#3498db', 
                label='Training score', linewidth=2)
        ax.fill_between(train_sizes_abs, 
                        train_mean - train_std, train_mean + train_std,
                        alpha=0.2, color='#3498db')
        
        ax.plot(train_sizes_abs, test_mean, 'o-', color='#e74c3c',
                label='Cross-validation score', linewidth=2)
        ax.fill_between(train_sizes_abs,
                        test_mean - test_std, test_mean + test_std,
                        alpha=0.2, color='#e74c3c')
        
        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title('Learning Curve - Model Performance vs Dataset Size', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add interpretation text
        final_gap = train_mean[-1] - test_mean[-1]
        if final_gap < 0.1:
            interpretation = "✅ Good: Low bias, low variance"
        elif final_gap > 0.2:
            interpretation = "⚠️  High variance (overfitting)"
        else:
            interpretation = "⚠️  Moderate variance"
        
        ax.text(0.05, 0.05, interpretation, transform=ax.transAxes,
                fontsize=11, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save:
            output_file = os.path.join(self.figures_dir, 'learning_curve.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_file}")
        
        plt.close()
        
        return train_sizes_abs, train_mean, test_mean
    
    def analyze_residuals(self, X_test, y_test, model=None, save=True):
        """
        Analyze prediction residuals for model diagnostics
        
        Args:
            X_test: Test features
            y_test: True test values
            model: Sklearn model
            save: Save figure
        """
        if model is None:
            model_path = os.path.join(self.models_dir, 'adrb2_rf_model.pkl')
            model = joblib.load(model_path)
        
        # Make predictions
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Predicted vs Actual
        axes[0, 0].scatter(y_test, y_pred, alpha=0.5, s=30)
        axes[0, 0].plot([y_test.min(), y_test.max()], 
                       [y_test.min(), y_test.max()], 
                       'r--', lw=2, label='Perfect prediction')
        axes[0, 0].set_xlabel('Actual pIC50')
        axes[0, 0].set_ylabel('Predicted pIC50')
        axes[0, 0].set_title('Predicted vs Actual Values')
        axes[0, 0].legend()
        
        # Add R² annotation
        r2 = r2_score(y_test, y_pred)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', 
                       transform=axes[0, 0].transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Residuals plot
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=30)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted pIC50')
        axes[0, 1].set_ylabel('Residuals (Actual - Predicted)')
        axes[0, 1].set_title('Residual Plot')
        
        # 3. Residuals distribution
        axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        
        # Add statistics
        mean_res = residuals.mean()
        std_res = residuals.std()
        axes[1, 0].text(0.05, 0.95, 
                       f'Mean: {mean_res:.4f}\nStd: {std_res:.4f}',
                       transform=axes[1, 0].transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4. Q-Q plot for normality check
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normality Check)')
        
        plt.tight_layout()
        
        if save:
            output_file = os.path.join(self.figures_dir, 'residual_analysis.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_file}")
        
        plt.close()
        
        # Statistical tests
        print("\n=== Residual Analysis ===")
        print(f"Mean residual: {mean_res:.6f} (should be ~0)")
        print(f"Std of residuals: {std_res:.4f}")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
        print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
        
        # Normality test
        _, p_value = stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)
        print(f"\nShapiro-Wilk normality test p-value: {p_value:.6f}")
        if p_value < 0.05:
            print("⚠️  Residuals may not be normally distributed")
        else:
            print("✅ Residuals appear normally distributed")
    
    def feature_importance_analysis(self, model=None, feature_names=None, top_n=20, save=True):
        """
        Analyze feature importance for Random Forest
        
        Args:
            model: Trained Random Forest model
            feature_names: Names of features (if None, uses indices)
            top_n: Number of top features to plot
            save: Save figure
        """
        if model is None:
            model_path = os.path.join(self.models_dir, 'adrb2_rf_model.pkl')
            model = joblib.load(model_path)
        
        if not hasattr(model, 'feature_importances_'):
            print("Model does not have feature_importances_ attribute")
            return
        
        importances = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'Bit_{i}' for i in range(len(importances))]
        
        # Get top features
        indices = np.argsort(importances)[::-1][:top_n]
        top_importances = importances[indices]
        top_names = [feature_names[i] for i in indices]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, top_n))
        ax.barh(range(top_n), top_importances, color=colors)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save:
            output_file = os.path.join(self.figures_dir, 'feature_importance.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_file}")
        
        plt.close()
        
        print(f"\n=== Top {top_n} Features ===")
        for i, (name, imp) in enumerate(zip(top_names, top_importances), 1):
            print(f"{i:2d}. {name}: {imp:.6f}")


def main():
    """
    Run comprehensive model evaluation
    """
    print("=" * 60)
    print("Advanced Model Evaluation & Diagnostics")
    print("=" * 60)
    
    evaluator = ModelEvaluator(project_dir='/home/claude/adrb2_discovery')
    
    # Load training data (you'll need to adapt this to your actual data loading)
    from main_pipeline import ADRB2DrugDiscovery
    
    pipeline = ADRB2DrugDiscovery(project_dir='/home/claude/adrb2_discovery')
    
    try:
        # Load data
        print("\n--- LOADING DATA ---\n")
        data = pipeline.load_and_prepare_data()
        
        X = data['X']
        y = data['y']
        
        # Split for testing
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Cross-validation
        print("\n--- CROSS-VALIDATION ---\n")
        cv_metrics, cv_results = evaluator.cross_validate_model(X, y, cv=5)
        
        # Learning curve
        print("\n--- LEARNING CURVE ---\n")
        evaluator.plot_learning_curve(X, y)
        
        # Residual analysis
        print("\n--- RESIDUAL ANALYSIS ---\n")
        evaluator.analyze_residuals(X_test, y_test)
        
        # Feature importance
        print("\n--- FEATURE IMPORTANCE ---\n")
        evaluator.feature_importance_analysis(top_n=20)
        
        print("\n" + "=" * 60)
        print("Evaluation Complete!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure training data is available")


if __name__ == "__main__":
    main()
