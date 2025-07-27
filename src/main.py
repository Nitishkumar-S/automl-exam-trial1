from automl.pipeline import AutoMLPipeline

def main():
    # Initialize the AutoML pipeline
    pipeline = AutoMLPipeline()

    # Load datasets
    datasets = pipeline.load_datasets(['data/dataset1', 'data/dataset2', 'data/dataset3'])

    # Start the training process
    pipeline.run_pipeline(datasets)

if __name__ == "__main__":
    main()