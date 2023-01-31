(GraphSAGE)
============


Requirements
------------

```bash
pip install requests torchmetrics
```




### Minibatch training for node classification

Train w/ mini-batch sampling in mixed mode (CPU+GPU) for node classification on "ogbn-products"
the following example shows the algorithm for node classification
the process is
1. take a mini-batch of training nodes and sample a subgraph from these nodes
2. gather node / edge features used by the subgraph from CPU memory and
   copy into GPU
3. GNN model forward & backward & weight updates

the following command is used to train ogbn-papers100M that can occupy the full GPU　so we must use minibatch to avoid out of memory (A100 has 40G　but the dataset is 50 G or so)

184s or so
```bash
python3 big_dataset_sample_on_CPU.py
tensorboard --logdir=/YOUR_PATH/log
# change your path depending on on_trace_ready=torch.profiler.tensorboard_trace_handler
```

this one will load the graph minibatch on the GPU after the function of _next_ in iterator 
179s or so
```bash
python3 big_dataset.py
tensorboard --logdir=/YOUR_PATH/log
# change your path depending on on_trace_ready=torch.profiler.tensorboard_trace_handler
```


the following command is used to train ogbn-products
```bash
python3 node_classification.py
tensorboard --logdir=/home/ycwang/GNN/log2 
# change your path depending on on_trace_ready=torch.profiler.tensorboard_trace_handler
```



the folowing command is used to run the algorithm that put the entire graph on the GPU before running training process
```
python3 node_classcification_variant.py
tensorboard --logdir=/home/ycwang/GNN/log
````

### Full graph training

Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
python3 train_full.py --dataset cora --gpu 0    # full graph
```

Results:
```
* cora: ~0.8330 
* citeseer: ~0.7110
* pubmed: ~0.7830
```
