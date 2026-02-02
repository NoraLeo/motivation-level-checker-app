"""
Model training module with MLflow experiment tracking.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd

logger = logging.getLogger(__name__)


class MoodModelTrainer:
    """Trains and evaluates mood classification models."""
    
    def __init__(self, experiment_name: str = "motivation-level-checker"):
        self.experiment_name = experiment_name
        self.vectorizer = None
        self.model = None
        
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
    
    def prepare_data(self, df: pd.DataFrame, text_column: str, label_column: str, test_size: float = 0.2):
        """Prepare train/test splits."""
        X = df[text_column].values
        y = df[label_column].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def train(
        self,
        X_train: List[str],
        y_train: List,
        X_test: List[str],
        y_test: List,
        max_features: int = 5000,
        C: float = 1.0
    ) -> Dict:
        """
        Train mood classification model with MLflow tracking.
        
        Args:
            X_train: Training texts
            y_train: Training labels
            X_test: Test texts
            y_test: Test labels
            max_features: Max features for TF-IDF
            C: Regularization parameter for logistic regression
        
        Returns:
            Dictionary with training metrics
        """
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("max_features", max_features)
            mlflow.log_param("C", C)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            
            # Vectorize text
            logger.info("Vectorizing text...")
            self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            # Train model
            logger.info("Training model...")
            self.model = LogisticRegression(C=C, max_iter=1000, random_state=42)
            self.model.fit(X_train_vec, y_train)
            
            # Make predictions
            y_pred_train = self.model.predict(X_train_vec)
            y_pred_test = self.model.predict(X_test_vec)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            train_f1 = f1_score(y_train, y_pred_train, average='weighted')
            test_f1 = f1_score(y_test, y_pred_test, average='weighted')
            
            # Log metrics
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("train_f1", train_f1)
            mlflow.log_metric("test_f1", test_f1)
            
            # Log model
            mlflow.sklearn.log_model(self.model, "model")
            
            logger.info(f"Training complete. Test accuracy: {test_accuracy:.4f}")
            logger.info(f"Classification Report:\n{classification_report(y_test, y_pred_test)}")
            
            return {
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "train_f1": train_f1,
                "test_f1": test_f1,
                "classification_report": classification_report(y_test, y_pred_test)
            }
    
    def save_model(self, model_path: str):
        """Save trained model and vectorizer to disk."""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer
            }, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load trained model and vectorizer from disk."""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.vectorizer = data['vectorizer']
        
        logger.info(f"Model loaded from {model_path}")
    
    def predict(self, texts: List[str]) -> List:
        """Make predictions on new texts."""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained or loaded")
        
        X_vec = self.vectorizer.transform(texts)
        return self.model.predict(X_vec)
