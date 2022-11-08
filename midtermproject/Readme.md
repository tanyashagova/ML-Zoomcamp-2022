This is a midterm project at ML-Zoomcamp-2022 cource. It is aimed to solve the employees attrition problem at the company. Model used here makes prediction of attrition of valuable employees. 

This project uses a fictional [dataset of employees attrition](https://www.kaggle.com/datasets/whenamancodes/hr-employee-attrition) created by IBM data scientists.

To run the project you can download the dataset from [Kaggle](https://www.kaggle.com/datasets/whenamancodes/hr-employee-attrition) or from the [project folder](https://github.com/tanyashagova/ML-Zoomcamp-2022/blob/main/midtermproject/HR%20Employee%20Attrition.csv) on GitHub.


## Description

Project folder contains

* Data 
* Notebook (`notebook.ipynb`) with data preparation, EDA, feature importance analysis and model selection process
* Script `train.py` which contains  training and saving the final model
* File `model.bin` with final model and dictvectorizer
* Script `predict.py` with model loading and serving it via a web serice (with Flask)
* `Pipenv` and `Pipenv.lock` files with dependencies
* `Dockerfile` for running the service
