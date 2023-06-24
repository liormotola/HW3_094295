# HW3_094295

## Goal
Train a GNN classification model on a citation network where nodes represent documents. Each node is described by a 128-dimensional feature vector.
Two documents are connected if there exists a citation link between them. The task is to infer the category of each document (40 in total).

## Description
In this project, we trained different models trying to achieve best accuracy score over the above task. 
<br>

This Reopsitory contains the following files:
 
<ul>

  * `exploration.py` - this holds the code of all data exploration we performed over the train data. 
  * `dataset.py` - loads the training dataset.
  * `train.py` - creating the different models we tested, performing the training procedure, and plots loss+accuracy graphs.
  * `Final_model.pkl` - our pre-trained model (which produced the best results over validation data).
  * `scaler.pkl` - a standard scaling object to use over the data.

 </ul>

## How to run?
#### Prepare envoriment
1. Clone this project
2. conda install the environment.yml file

### Reproduce results
#### predict.py
This will use our best pre-trained model - `Final_model.pkl` to predict the article's category over new data.<br>

