"""
Training script for mood classification model.
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
from motivation_checker.models.train import MoodModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train mood classification model")
    parser.add_argument("--data-path", type=str, required=True, help="Path to training data CSV")
    parser.add_argument("--text-column", type=str, default="text", help="Name of text column")
    parser.add_argument("--label-column", type=str, default="label", help="Name of label column")
    parser.add_argument("--output-path", type=str, default="data/models/mood_classifier.pkl", help="Output model path")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size")
    parser.add_argument("--max-features", type=int, default=5000, help="Max TF-IDF features")
    parser.add_argument("--C", type=float, default=1.0, help="Regularization parameter")
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Initialize trainer
    trainer = MoodModelTrainer()
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        df, args.text_column, args.label_column, args.test_size
    )
    
    # Train model
    logger.info("Training model...")
    metrics = trainer.train(X_train, y_train, X_test, y_test, args.max_features, args.C)
    
    # Save model
    trainer.save_model(args.output_path)
    logger.info(f"Model saved to {args.output_path}")
    logger.info(f"Test accuracy: {metrics['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
