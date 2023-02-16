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
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    sampler = NeighborSampler([15, 10, 5],  # fanout for [layer-0, layer-1, layer-2]
                              prefetch_node_feats=['feat'],
                              prefetch_labels=['label'])
    use_uva = (args.mode == 'mixed')
    train_dataloader = DataLoader(g, train_idx, sampler, device=device,
                                  batch_size=1024, shuffle=True,
                                  drop_last=False, num_workers=0,
                                  use_uva=use_uva
                                  )

    val_dataloader = DataLoader(g, val_idx, sampler, device=device,
                                batch_size=1024, shuffle=True,
                                drop_last=False, num_workers=0,
                                use_uva=use_uva
                                )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    for epoch in range(1):
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            print('it: ', it)
            if it > 20:
                break
            ## consider block device
            # blocks = [b.to(device) for b in blocks]
            # import pdb; pdb.set_trace()
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label']
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        acc = evaluate(model, g, val_dataloader)
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} "
              .format(epoch, total_loss / (it+1), acc.item()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='mixed', choices=['cpu', 'mixed', 'puregpu'],
                        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                             "'puregpu' for pure-GPU training.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = 'cpu'
    print(f'Training in {args.mode} mode.')

    # load and preprocess dataset
    print('Loading data')
    print(args.mode)
    dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-arxiv'))
    g = dataset[0]
    g = g.to('cuda' if args.mode == 'puregpu' else 'cpu')
    device = torch.device('cpu' if args.mode == 'cpu' else 'cuda')

    # create GraphSAGE model
    # import pdb; pdb.set_trace()
    in_size = g.ndata['feat'].shape[1]
    out_size = dataset.num_classes
    print('in_size', in_size)
    print('out_size', out_size)
    model = SAGE(in_size, 256, out_size).to(device)
    print('device: ', device)
    # print('model device: ', model.device)

    # model training
    print('Training...')
    
    proflie=True
    if proflie==True:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            profile_memory=True,
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./reproduce_gpu/arxiv_1024'),
            record_shapes=True,
            with_stack=True)
        prof.start()
        # for step, batch_data in enumerate(train_loader):
        #     if step >= (1 + 1 + 3) * 2:
        #         break
        #     train(batch_data)
        #     prof.step()
        train(args, device, g, dataset, model)
        prof.stop()
    else:
        train(args, device, g, dataset, model)
    

    # test the model
    print('Testing...')
    acc = layerwise_infer(device, g, dataset.test_idx, model, batch_size=4096)
    print("Test Accuracy {:.4f}".format(acc.item()))
