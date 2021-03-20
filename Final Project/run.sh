rm -rf experiment_data
jupyter nbconvert --to script cycle_dataset.ipynb
jupyter nbconvert --to script experiment.ipynb
jupyter nbconvert --to script dataset_factory.ipynb
