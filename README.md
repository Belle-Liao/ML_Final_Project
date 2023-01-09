# ML_Final_Project
2023 NYCU Introduction to Machine Learning, Final Project

## Requirements 

The local environmet is VS code.

Here I provide the version of all packets importing in the code:
```python
numpy -- 1.22.2
panda -- 1.3.0
sklearn -- 1.0.2
lightgbm -- 3.3.4
catboost -- 1.1.1
```
    
## Training

The [training code](https://github.com/Belle-Liao/ML_Final_Project/blob/main/109550024_Final_train.ipynb) is a jupyter notebook (.ipynb). 

In VS code, just **Run All** the cells and you can start the training process.

## Evaluation

The [inference code](https://github.com/Belle-Liao/ML_Final_Project/blob/main/109550024_Final_inference.ipynb) is a jupyter notebook (.ipynb). 

Downloading the [my model](https://github.com/Belle-Liao/ML_Final_Project/blob/main/LR_model.pkl) first. Then put it in the same file of the inference code and just **Run All** the cells in the VS code

### Load the Model
```python
import pickle

pkl_filename="LR_model.pkl"
with open(pkl_filename, 'rb') as file:
    pickled_LR_model = pickle.load(file)
```

### Reproduce the submission csv file
```python
test_predict = pickled_LR_model.predict_proba(X_test)[:,1]
sample_submission['failure'] = test_predict
sample_submission.to_csv('109550024.csv', index=False)
```

## Model Link

I provide two links below but they are the same model:
* [Github](https://github.com/Belle-Liao/ML_Final_Project/blob/main/LR_model.pkl)
* [Google Cloud](https://drive.google.com/drive/u/0/folders/12v9viRiIVK6zNRVC9tQmDNxmddY96csd)

## Reproduce the Submission

After **Run All** all the cells, a csv file **109550024.csv** will appear in the same file of the inference code.

## Results

Competiton on Kaggle: [Tabular Playground Series - Aug 2022](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/overview)
Baseline: 0.58990 on private score

My best result by using the LogisticRegression model:
| Model Name   | Private Score | Public Score |
| ------------ | ------------- | ------------ |
| LR_model.pkl | 0.5909        | 0.59114      |

