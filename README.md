# EECE6822Project

This repository contains all data and source code used by Matthew Dupont for his EECE6822 Project, "Day-Ahead Global Horizontal Irradiance Forecasting using LSTM Deep Learning Models". 



 `conda` was used as a combination package and environment manager. It can be installed from here: https://docs.conda.io/en/latest/miniconda.html Once `conda` is installed, either add `conda` integration to PowerShell with the command `conda init powershell` . I haven't tried loading this project on a Mac or Unix machine, so I'm unable to help use conda on those machines. Once conda is installed, you can create a copy of the environment used during project development from the attached environment.yml file with the command `conda env create -f environment.yml`. This should create an environment `solar_forecasting_project` using the appropriate version of the Python interpreter and including all dependencies used during project development. Then activate this environment with `conda activate solar_forecasting_project`. From this environment, Python calls may be run to test the provided code, and will automatically use the python interpreter of the environment and include all dependencies. From an IDE such as Visual Studio Code, features like the "Run Code" option, as well as Jupyter Notebooks, will require selecting an interpreter - to use these features, select the interpreter associated with the `conda` environment.

`conda` may have some adverse interactions with `pip`, especially when executing Python code. If you encounter issues with these, ~~God help you~~ please feel free to contact the author for help :).



Folders exist as follows:

.

|-- data_cleaned 

​	- Contains preprocessed data used for model training.

|-- logs 

	- Log messages from the most recent batch of models, those used for the project. 
	- <>_summary files detail a summary of model hyperparameters.
	- <>_log files capture the text output emitted during model training.

|-- Misc_models

​	- Some stored model files from early training.

|-- models

	- Model files generated from training. Can be reloaded from tensorflow for future use during evaluation.

|-- src

 - Source code files used to run models, preprocess data, etc. Main files are:
 - solar_forecasting_model.py - main call to generate all models.
 - models.py - multiple variations on slight tweaks to model architecture.
 - generate_features.py - code to split input data into features, expand features into history/predictions, and generate harmonic terms.
 - clean_data.py - data preprocessing, cleaning, etc.
 - constants.py - hard-coded constants used throughout the code. Note that use here may be inconsistent - not the cleanest code practice on my part. 
 - accumulated_results.csv - csv log of each station, featureset, and iteration, with aggregate measures.
 - single_forecast.py and test_model.py were used during exploration of tensorflow features, and not included in the final paper.

Notable files include `best_results.csv` and `results_melted.csv`, storing intermediate results for display in tables. Also particularly important is `test_notebook.ipynb` - this Jupyter notebook was used for various testing, and in particular to generate many of the visualizations used in this project.

