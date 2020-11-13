# Project 2 - FYS-STK3155

## Author
* _Name_: Emilie Fjørner
* _Email_: emiliefj@fys.uio.no

Folder for my work on project 2 in the course fys-stk3155 at UiO fall of 2020.

## Structure

### Code
A folder containing the code for the models usen in this project, that is
* Linear regression with ridge parametrization and stochastic gradient descent
* Logistic regression (multimomial with softmax)
* Feed forward neural network for regression
* Feed forward neural network for classification

### Data
Contains data that can be used for testing the models. None of these are actually used in the final report, though I have tested my FFNN code on the mnist data and gotten good results.
The data actually used in the report are a subset of the mnist data available through scikit-learn as sklearn.datasets.load_digits(), as well as data generated using the Franke function from project 1

### Report
Contains the final report as well as a jupyter notebook containing several code snippits to try out the code, with expected results.

### Results
A selection of plots showing model performance and parameter selection.