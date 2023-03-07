import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
import argparse
from numpy import *

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'mean'))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # import pdb; pdb.set_trace()
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata['feat']
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False,
                num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(), self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device, pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x) # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0]:output_nodes[-1]+1] = h.to(buffer_device)
            feat = y
        return y

def evaluate(model, graph, dataloader):
    dataloader.device = device
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata['feat']
            ys.append(blocks[-1].dstdata['label'])
            y_hats.append(model(blocks, x))
    return MF.accuracy(torch.cat(y_hats), torch.cat(ys),task='multiclass',num_classes=out_size)

def layerwise_infer(device, graph, nid, model, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size) # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata['label'][nid].to(pred.device)
        return MF.accuracy(pred, label,task='multiclass',num_classes=out_size)

def train(args, device, g, dataset, model):
    
   
    
    # create sampler & dataloader
    cpu_device=torch.device('cpu')
    # print('dataset device: ', dataset.train_idx.device)
    train_idx = dataset.train_idx.to(cpu_device)
    val_idx = dataset.val_idx.to(cpu_device)
    sampler = NeighborSampler([15, 10, 5])
    use_uva = (args.mode == 'mixed')
    pin_prefetcher_ = True
    train_dataloader = DataLoader(g, train_idx, sampler, device=cpu_device,
                                  batch_size=1024, shuffle=True,
                                  drop_last=False, num_workers=0,
                                #   pin_prefetcher=pin_prefetcher_,
                                #   use_uva=False
                                  )
    ## 2 stand for using cuda event 3 stand for using time.time
    train_dataloader.whether_time_time_cudaevent =whether_time_time_cudaevent

    val_dataloader = DataLoader(g, val_idx, sampler, device=cpu_device,
                                batch_size=1024, shuffle=True,
                                drop_last=False, num_workers=0,
                                # pin_prefetcher=pin_prefetcher_,
                                # use_uva=False
                                )
    
    val_dataloader.whether_time_time_cudaevent =whether_time_time_cudaevent
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    
    for epoch in range(1):
        model.train()
        total_loss = 0
        import time
        
        
        if train_dataloader.whether_time_time_cudaevent == 3:
            torch.cuda.synchronize()
            start_time = time.time()
        elif train_dataloader.whether_time_time_cudaevent == 2:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record(torch.cuda.current_stream())
            
        batch_prepare_time =[]
        data_tansfer_time = []
        train_time = []
        slice_time = []
        feature_data_trans=[]
        block_pin=[]
        
        def profile_time(profile_list,add_last_time=False):
            nonlocal start_time
            nonlocal start_event
            nonlocal end_event
            if train_dataloader.whether_time_time_cudaevent == 3:
                torch.cuda.synchronize()
                end_time = time.time()
                if add_last_time==True:
                    profile_list[-1]+=((end_time-start_time)*1000)
                else:
                    profile_list.append((end_time-start_time)*1000)
                torch.cuda.synchronize()
                start_time = time.time()
            elif train_dataloader.whether_time_time_cudaevent == 2:
                end_event.record(torch.cuda.current_stream())
                end_event.synchronize()
                step_elapsed = start_event.elapsed_time(end_event) 
                if add_last_time==True:
                    profile_list[-1]+=step_elapsed
                else:
                    profile_list.append(step_elapsed)
                torch.cuda.synchronize()
                start_event.record(torch.cuda.current_stream())
        
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            print('it: ', it)
            
            if it> 50:
                break
            
            profile_time(batch_prepare_time)
            # blocks = [b.pin_memory_() for b in blocks]
            profile_time(block_pin)
            
            block0_id=blocks[0].srcdata['_ID']
            block_last_id=blocks[-1].dstdata['_ID']
            blocks = [b.to(device) for b in blocks]
            profile_time(data_tansfer_time)
            profile_fine_grained = False
            if profile_fine_grained==True:
                x, dgl_slice_time, dgl_feature_trans_time = blocks[0].srcdata['feat']
                slice_time.append(dgl_slice_time)
                feature_data_trans.append(dgl_feature_trans_time)
                if paper_100m==True:
                    y, dgl_slice_time, dgl_feature_trans_time = blocks[-1].dstdata['label']
                    y=y.to(torch.int64)
                    slice_time[-1]+=dgl_slice_time
                    feature_data_trans[-1]+=dgl_feature_trans_time
                else:
                    y , dgl_slice_time, dgl_feature_trans_time= blocks[-1].dstdata['label']
                    slice_time[-1]+=dgl_slice_time
                    feature_data_trans[-1]+=dgl_feature_trans_time
            else:
                x = torch.empty(block0_id.shape[0],in_size, pin_memory=True)
                torch.index_select(train_dataloader.graph.ndata['feat'], 0, block0_id,out=x)
                profile_time(slice_time)
                x=x.to(device)
                profile_time(feature_data_trans)
                
                # x= blocks[0].srcdata['feat']
                if paper_100m==True:
                    y = blocks[-1].dstdata['label'].to(torch.int64)
                else:
                    # y= blocks[-1].dstdata['label']
                    
                    y = torch.empty(block_last_id.shape[0], pin_memory=True).to(torch.int64)
                    torch.index_select(train_dataloader.graph.ndata['label'], 0, block_last_id,out=y)
                    profile_time(slice_time,add_last_time=True)
                    y=y.to(device)
                    profile_time(feature_data_trans,add_last_time=True)
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            profile_time(train_time)
            
        ## calculate the avg time of a list
        batch_prepare_time_avg = mean(batch_prepare_time[-15:-1])
        data_tansfer_time_avg = mean(data_tansfer_time[-15:-1])
        train_time_avg = mean(train_time[-15:-1])
        slice_time_avg = mean(slice_time[-15:-1])
        feature_data_transition_time_avg = mean(feature_data_trans[-15:-1])
        block_pin_avg = mean(block_pin[-15:-1])
        all_avg = batch_prepare_time_avg + data_tansfer_time_avg + train_time_avg+slice_time_avg+feature_data_transition_time_avg+block_pin_avg
        
        print('batch_prepare_time: ', batch_prepare_time)
        print('block_pin: ', block_pin)
        print('data_tansfer_time: ', data_tansfer_time)
        print('slice_time: ', slice_time)
        print('feature slice',feature_data_trans)
        print('train_time: ', train_time)
        
        print('batch_prepare_time_avg: ', batch_prepare_time_avg)
        print('block_pin_avg: ', block_pin_avg)
        print('data_tansfer_time_avg: ', data_tansfer_time_avg)
        print('slice_time_avg: ', slice_time_avg)
        print('feature_data_transition_time_avg: ', feature_data_transition_time_avg)
        print('train_time_avg: ', train_time_avg)
        print('all_avg: ', all_avg)        
        
        acc = evaluate(model, g, val_dataloader)
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} "
              .format(epoch, total_loss / (it+1), acc.item()))
        exit(0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='mixed', choices=['cpu', 'mixed', 'puregpu'],
                        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                             "'puregpu' for pure-GPU training.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = 'cpu'
    print(f'Training in {args.mode} mode.')
    
    ## 2 stand for using cuda event 3 stand for using time.time do not need to break down
    whether_time_time_cudaevent = 3
    print('whether_time_time_cudaevent: ', whether_time_time_cudaevent)
    
    import os

    filename = os.path.basename(__file__)
    print(filename)

    # load and preprocess dataset
    print('Loading data')
    print(args.mode)
    dataset_name='ogbn-arxiv'
    dataset = AsNodePredDataset(DglNodePropPredDataset(dataset_name))
    print('dataset: ', dataset_name)
    g = dataset[0]
    g = g.to('cuda' if args.mode == 'puregpu' else 'cpu')
    device = torch.device('cpu' if args.mode == 'cpu' else 'cuda')

    # create GraphSAGE model
    in_size = g.ndata['feat'].shape[1]
    out_size = dataset.num_classes
    print('in_size', in_size)
    print('out_size', out_size)
    paper_100m=False
    if out_size==172:
        paper_100m=True
    model = SAGE(in_size, 256, out_size).to(device)
    print('device: ', device)
    # print('model device: ', model.device)

    # model training
    print('Training...')
    
    proflie=True
    train(args, device, g, dataset, model)
    

    # test the model
    print('Testing...')
    acc = layerwise_infer(device, g, dataset.test_idx, model, batch_size=4096)
    print("Test Accuracy {:.4f}".format(acc.item()))
