# my_pipgcn
My implementation of the 2017 NIPS paper titled [Protein Interface Prediction using Graph Convolutional Networks](https://papers.nips.cc/paper/7231-protein-interface-prediction-using-graph-convolutional-networks).

The code in this repository wil reproduce the results described in Table 2 for the following methods:

1. Node Average (Equation 1)
2. Node and Edge Average (Equation 2)
3. Order Dependent (Equation 3)

## Setup
### Requirements
* Python 3.6.3
* TensorFlow 1.4.0
* scikit-learn 0.19.1

## Usage

The scripts to train and evaluate the Node Average (Equation 1), Node and Edge Average (Equation 2) and Order Dependent (Equation 3) models are in `src/node_average.py`, `src/node_and_edge_average.py` and `src/order_dependent.py` respectively. The scripts will save the results into the `/model_dir` directory based on the values in the `params` dictionary. The saved results consist of the model's graph, last 2 checkpoints and a csv of the ROC AUC on the test set (filename: eperiment_summary.csv).

## Results

I had enough GPU credits to only run the Node Average (Equation 1) 1 layer model once (in the paper, each model was run 10 times and the average ROC AUC over the 10 trials was reported). The results are saved in `/model_dir/node_average/node_avg_1_layer`. The ROC AUC from this run is 0.854 vs the paper's 0.864. 



