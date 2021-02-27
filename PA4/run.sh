rm -rf experiment_data
echo Cleared
jupyter nbconvert --to script lstm.ipynb
jupyter nbconvert --to script edit_experiment.ipynb
echo Generated completed
python main.py
