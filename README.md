GraphSAGE
============


Requirements
------------

```bash
pip install requests torchmetrics
```


## baseline result

<!-- ![alt 属性文本]( https://files.slack.com/files-pri/TR41NUQES-F04PV6FCFSR/image.png)


<img src="https://files.slack.com/files-pri/TR41NUQES-F04PV6FCFSR/image.png" alt="图片alt" title="图片title">



![CSDN图标](https://csdnimg.cn/cdn/content-toolbar/csdn-logo_.png?v=20190924.1 "CSDN图标") -->

![image.png](https://s2.loli.net/2023/02/16/QaxvnSr31jgzuDB.png)



### configuration 
batch size:1024,
neighbor size:[15, 10, 5],  # fanout for [layer-0, layer-1, layer-2],
number of layers:3 

#### sampled subgraph size compared to the whole graph：

arxiv: 0.15153859326928187

> len(dataset.train_idx)=169343
0:Block(num_src_nodes=25662, num_dst_nodes=10752, num_edges=36362)
1:
Block(num_src_nodes=10752, num_dst_nodes=3611, num_edges=11181)
2:
Block(num_src_nodes=3611, num_dst_nodes=1024, num_edges=2641)

products : 0.18791365884193287

> all_node=2449029
0:Block(num_src_nodes=460206, num_dst_nodes=57939, num_edges=850617)
1:
Block(num_src_nodes=57939, num_dst_nodes=6073, num_edges=60159)
2:
Block(num_src_nodes=6073, num_dst_nodes=1024, num_edges=5114)



core code
```
train_dataloader = DataLoader(g, train_idx, sampler, device=cpu_device,
                                  batch_size=1024, shuffle=True,
                                  drop_last=False, num_workers=0,
                                  )

val_dataloader = DataLoader(g, val_idx, sampler, device=cpu_device,
                           batch_size=1024, shuffle=True,
                           drop_last=False, num_workers=0,
                           )
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
for epoch in range(1):
   model.train()
   total_loss = 0
   for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
      if it> 20 and paper_100m==False:
            break
      if it> 3 and paper_100m==True:
            break
      blocks = [b.to(device) for b in blocks]
      # import pdb; pdb.set_trace()
      x = blocks[0].srcdata['feat']
      if paper_100m==True:
            y = blocks[-1].dstdata['label'].to(torch.int64)
      else:
            y = blocks[-1].dstdata['label']
      y_hat = model(blocks, x)
      loss = F.cross_entropy(y_hat, y)
      opt.zero_grad()
      loss.backward()
      opt.step()
      total_loss += loss.item()
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
python3 big_dataset.py
tensorboard --logdir=/YOUR_PATH/log
# change your path depending on on_trace_ready=torch.profiler.tensorboard_trace_handler
```

this one will load the graph minibatch on the GPU after the function of _next_ in iterator 
179s or so
```bash
python3 big_dataset_sample_on_CPU.py
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
