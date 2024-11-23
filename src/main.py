import os
import argparse
import logging
import yaml
import torch
from datetime import datetime
import pandas as pd

from data.binance_fetcher import BinanceFetcher
from features.feature_engineering import FeatureEngineer
from models.base_model import BaseModel
from training.trainer import Trainer
from evaluation.evaluator import Evaluator

def setup_logging(log_dir: str) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
        
    Returns:
        Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('bitcoin_trading')
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(
        os.path.join(log_dir, f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    )
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def download_data(config: dict, logger: logging.Logger) -> None:
    """
    Download historical data from Binance.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("Starting data download...")
    
    fetcher = BinanceFetcher(
        symbol=config['data']['binance']['symbol'],
        interval=config['data']['binance']['interval'],
        start_date=config['data']['binance']['start_date'],
        end_date=config['data']['binance']['end_date']
    )
    
    fetcher.fetch_data()
    logger.info("Data download completed.")

def process_features(config: dict, logger: logging.Logger) -> None:
    """
    Process and engineer features.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("Starting feature engineering...")
    
    # Load raw data
    raw_data_path = os.path.join(
        'data/raw',
        f"{config['data']['binance']['symbol']}_{config['data']['binance']['interval']}_data.parquet"
    )
    
    engineer = FeatureEngineer(
        input_file=raw_data_path,
        output_dir='data/processed',
        config=config['data']['features']
    )
    
    engineer.generate_features()
    logger.info("Feature engineering completed.")

def prepare_datasets(processed_data_path: str, split_ratios: list = [0.7, 0.15, 0.15]) -> tuple:
    """
    Prepare train, validation, and test datasets.
    
    Args:
        processed_data_path: Path to processed data
        split_ratios: List of ratios for train/val/test splits
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Load processed data
    data = pd.read_parquet(processed_data_path)
    
    # Sort by time
    data = data.sort_values('open_time')
    
    # Calculate split points
    train_end = int(len(data) * split_ratios[0])
    val_end = int(len(data) * (split_ratios[0] + split_ratios[1]))
    
    # Split data
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]
    
    return train_data, val_data, test_data

def train_model(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    config: dict,
    logger: logging.Logger
) -> BaseModel:
    """
    Train the trading model.
    
    Args:
        train_data: Training data
        val_data: Validation data
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Trained model
    """
    logger.info("Starting model training...")
    
    # Create model
    model = BaseModel(
        state_dim=train_data.shape[1],
        action_dim=1,
        hidden_dim=config['model']['architecture']['actor_hidden_layers'][0],
        num_heads=config['model']['architecture']['attention_heads']
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        config=config,
        experiment_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Train model
    trainer.train(
        num_episodes=1000,
        validate_every=10,
        save_every=100
    )
    
    logger.info("Model training completed.")
    return model

def evaluate_model(
    model: BaseModel,
    test_data: pd.DataFrame,
    config: dict,
    logger: logging.Logger
) -> None:
    """
    Evaluate the trained model.
    
    Args:
        model: Trained model
        test_data: Test data
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("Starting model evaluation...")
    
    evaluator = Evaluator(
        model=model,
        test_data=test_data,
        config=config
    )
    
    metrics = evaluator.backtest()
    
    logger.info("Evaluation metrics:")
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.4f}")
    
    logger.info("Model evaluation completed.")

def main():
    """
    Main execution function.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Bitcoin Trading RL')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['full', 'train', 'evaluate'],
                      default='full', help='Execution mode')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging('results/logs')
    logger.info("Starting Bitcoin Trading RL pipeline...")
    
    try:
        if args.mode in ['full']:
            # Download data
            download_data(config, logger)
            
            # Process features
            process_features(config, logger)
        
        # Prepare datasets
        processed_data_path = os.path.join(
            'data/processed',
            f"features_{datetime.now().strftime('%Y%m%d')}.parquet"
        )
        train_data, val_data, test_data = prepare_datasets(processed_data_path)
        
        if args.mode in ['full', 'train']:
            # Train model
            model = train_model(train_data, val_data, config, logger)
        else:
            # Load model for evaluation
            model = BaseModel(
                state_dim=test_data.shape[1],
                action_dim=1,
                hidden_dim=config['model']['architecture']['actor_hidden_layers'][0],
                num_heads=config['model']['architecture']['attention_heads']
            )
            model.load('results/checkpoints/best_model.pt')
        
        if args.mode in ['full', 'evaluate']:
            # Evaluate model
            evaluate_model(model, test_data, config, logger)
        
        logger.info("Pipeline completed successfully.")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
