import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "SIBI_classifier"

list_of_files = [
    # Core directory placeholder for Git to recognize empty folders. Helps in initializing version control even if folders are empty.
    "data/.gitkeep",  
    
    # Base module initialization for the project. Ensures Python treats these directories as modules and supports modular development.
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",

    # DATA INGESTION COMPONENTS. Manages the collection of raw data from various sources. Includes utils for reusable ingestion methods.
    f"src/{project_name}/components/data_ingestion_components/__init__.py",
    f"src/{project_name}/components/data_ingestion_components/data_ingestion.py",
    f"src/{project_name}/components/data_ingestion_components/utils/__init__.py",

    # DATA PREPROCESSING COMPONENTS. Handles cleaning, transformation, and preparation of data for training.
    f"src/{project_name}/components/data_preprocessing_components/__init__.py",
    f"src/{project_name}/components/data_preprocessing_components/data_preprocessing.py",
    f"src/{project_name}/components/data_preprocessing_components/utils/__init__.py",

    # MODEL TRAINER COMPONENTS. Contains logic for training machine learning models and saving results.
    f"src/{project_name}/components/model_trainer_components/__init__.py",
    f"src/{project_name}/components/model_trainer_components/model_trainer.py",
    f"src/{project_name}/components/model_trainer_components/utils/__init__.py",

    # MODEL EVALUATION COMPONENTS. Evaluates trained models for accuracy and performance metrics.
    f"src/{project_name}/components/model_evaluation_components/__init__.py",
    f"src/{project_name}/components/model_evaluation_components/model_evaluation.py",
    f"src/{project_name}/components/model_evaluation_components/utils/__init__.py",

    # MODEL PUSHER COMPONENTS. Responsible for pushing trained models to production or storage.
    f"src/{project_name}/components/model_pusher_components/__init__.py",
    f"src/{project_name}/components/model_pusher_components/model_pusher.py",
    f"src/{project_name}/components/model_pusher_components/utils/__init__.py",

    # PREDICTION COMPONENTS. Handles real-time predictions or batch inference based on trained models.
    f"src/{project_name}/components/prediction_components/__init__.py",
    f"src/{project_name}/components/prediction_components/prediction.py",
    f"src/{project_name}/components/prediction_components/utils/__init__.py",

    # HYPERPARAMETER TUNING COMPONENTS. Optimizes model parameters to improve performance.
    f"src/{project_name}/components/hyperparameter_tuning_components/__init__.py",
    f"src/{project_name}/components/hyperparameter_tuning_components/hyperparameter_tuning.py",
    f"src/{project_name}/components/hyperparameter_tuning_components/utils/__init__.py",

    # CONFIGURATION. Stores and manages project configurations (paths, hyperparameters, etc.).
    f"src/{project_name}/configuration/__init__.py",
    f"src/{project_name}/configuration/configuration.py",

    # CONSTANT. Defines constants used across the project to maintain consistency.
    f"src/{project_name}/constant/__init__.py",

    # ENTITY. Defines data structures or classes representing input/output entities.
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",

    # EXCEPTION. Provides custom exception handling for better error management.
    f"src/{project_name}/exception/__init__.py",

    # LOGGER. Manages logging for tracking application behavior and debugging.
    f"src/{project_name}/logger/__init__.py",
    f"src/{project_name}/logger/logger_config.py",

    # PIPELINE. Contains the orchestrated workflows for training, prediction, and hyperparameter tuning.
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/pipeline/prediction_pipeline.py",
    f"src/{project_name}/pipeline/hyperparameter_tuning_pipeline.py",

    # Utility modules for shared functions. Reusable methods like file management or metric calculations.
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/main_utils.py",

    # Machine Learning-specific modules. Contains ML model.
    f"src/{project_name}/ml/__init__.py",
    f"src/{project_name}/ml/model.py",

    # HTML template for the web application. Provides structure for visualizing predictions or data.
    "template/index.html",

    # Excludes unnecessary files from Docker images.  Optimizes Docker images for deployment.
    ".dockerignore",

    # Main application script. Entry point for running the project (e.g., REST API or CLI).
    "app.py",

    # Dockerfile for containerizing the project. Allows deployment in isolated and consistent environments.
    "Dockerfile",

    # Lists dependencies required to run the project. Ensures reproducibility across environments.
    "requirements.txt",

    # Script for installing the package. Enables project distribution as a Python package.
    "setup.py",

    # YAML file for storing hyperparameters and other configs. Facilitates easy parameter adjustments without modifying code.
    "params.yaml",
    "config.yaml",

    # Script for visualizing the directory structure. Useful for understanding the project's organization.
    "dir_tree_structure.py",

    "notebooks/trials.ipynb",

    ".env"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # Buat direktori jika belum ada
    if filedir:
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    # Cek jika file belum ada, baru buat file kosong
    if not filepath.exists():
        with open(filepath, 'w') as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"File {filepath} already exists and will not be overwritten.")