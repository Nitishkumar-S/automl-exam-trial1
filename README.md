# AutoML Image Modality Project

This project implements an AutoML pipeline for image modality datasets using learning curve extrapolation and Bayesian optimization for hyperparameter tuning of a neural network model.

## Project Structure

```
automl-image-modality
├── src
│   ├── main.py                # Entry point for the AutoML pipeline
│   ├── automl
│   │   ├── pipeline.py        # Orchestrates the AutoML process
│   │   ├── bayesian_optimization.py # Implements Bayesian Optimization
│   │   └── learning_curve.py  # Extrapolates learning curves
│   ├── models
│   │   └── neural_network.py   # Defines the neural network architecture
│   ├── data
│   │   ├── dataset1
│   │   │   └── __init__.py    # Loads and preprocesses dataset1
│   │   ├── dataset2
│   │   │   └── __init__.py    # Loads and preprocesses dataset2
│   │   └── dataset3
│   │       └── __init__.py    # Loads and preprocesses dataset3
│   └── utils
│       └── helpers.py         # Utility functions for the project
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Files to ignore in version control
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd automl-image-modality
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the AutoML pipeline, execute the following command:
```
python src/main.py
```

This will initialize the pipeline, load the datasets, and start the training process.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.
