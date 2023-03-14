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
import contextlib
import time
import numpy


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
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
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


@contextlib.contextmanager
def with_profile_time(records, whether_time_time_cudaevent=2):
    if whether_time_time_cudaevent == 3:
        torch.cuda.synchronize()
        start = time.time()
        yield
        torch.cuda.synchronize()
        end = time.time()
        records.append((end - start) * 1000)
    elif whether_time_time_cudaevent == 2:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record(torch.cuda.current_stream())
        yield
        end_event.record(torch.cuda.current_stream())
        end_event.synchronize()
        step_elapsed = start_event.elapsed_time(end_event)
        records.append(step_elapsed)


def evaluate(model, graph, dataloader):
    dataloader.device = device
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    # import pdb; pdb.set_trace()
    return MF.accuracy(
        torch.cat(y_hats), torch.cat(ys), task="multiclass", num_classes=out_size
    )


def layerwise_infer(device, graph, nid, model, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size)  # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)
        return MF.accuracy(pred, label, task="multiclass", num_classes=out_size)


def train(args, device, g, dataset, model):
    random.seed(1)
    numpy.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    dgl.seed(1)
    ##TODO
    global profile_in_frame
    profile_in_frame = True
    # create sampler & dataloader
    cpu_device = torch.device("cpu")
    # print('dataset device: ', dataset.train_idx.device)
    train_idx = dataset.train_idx.to(cpu_device)
    val_idx = dataset.val_idx.to(cpu_device)
    sampler = NeighborSampler([15, 10, 5])
    use_uva = args.mode == "mixed"
    pin_prefetcher_ = True
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=cpu_device,
        batch_size=1024,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    ## 2 stand for using cuda event 3 stand for using time.time
    train_dataloader.whether_time_time_cudaevent = whether_time_time_cudaevent

    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=cpu_device,
        batch_size=1024,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        # pin_prefetcher=pin_prefetcher_,
        # use_uva=False
    )

    val_dataloader.whether_time_time_cudaevent = whether_time_time_cudaevent
    train_dataloader.graph.ndata["label"] = train_dataloader.graph.ndata["label"].to(
        torch.int64
    )
    if args.dataset=='yelp':
        train_dataloader.graph.ndata["label"] = train_dataloader.graph.ndata["label"].to(
        torch.float64
    )
    train_dataloader.graph.ndata["feat"] = train_dataloader.graph.ndata["feat"].pin_memory()
    train_dataloader.graph.ndata["label"] = train_dataloader.graph.ndata["label"].pin_memory()
    
    # import pdb; pdb.set_trace()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    avg_epoch_batch_prepare_time = []
    avg_epoch_data_tansfer_time = []
    avg_epoch_train_time = []
    avg_epoch_slice_time = []
    avg_epoch_feature_data_trans = []
    avg_epoch_all_time = []
    for epoch in range(5):
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
        batch_prepare_time = []
        data_tansfer_time = []
        train_time = []
        slice_time = []
        feature_data_trans = []
        block_pin = []
        allocate_slice_time=[]
        it=0
        while True:
            if it > 100:
                break
            with with_profile_time(batch_prepare_time,whether_time_time_cudaevent=whether_time_time_cudaevent):
                (input_nodes, output_nodes, blocks) =next(iter(train_dataloader))
            if epoch == 0 and it==1:
                print('blocks: ', blocks)
            if it%50==0:
                print("it: ", it)
            it+=1
            
            with with_profile_time(data_tansfer_time, whether_time_time_cudaevent=whether_time_time_cudaevent):
                block0_id = blocks[0].srcdata["_ID"]
                block_last_id = blocks[-1].dstdata["_ID"]
                blocks = [b.to(device) for b in blocks]
            with with_profile_time(slice_time, whether_time_time_cudaevent=whether_time_time_cudaevent):
                x = torch.empty(block0_id.shape[0], in_size, pin_memory=True)
                if args.dataset=='yelp':
                    y = torch.empty((block_last_id.shape[0],100), pin_memory=True, dtype=torch.float64)
                else:
                    y = torch.empty(block_last_id.shape[0], pin_memory=True, dtype=torch.int64)
                torch.index_select(
                    train_dataloader.graph.ndata["feat"], 0, block0_id, out=x
                )
                torch.index_select(
                    train_dataloader.graph.ndata["label"], 0, block_last_id, out=y
                )
            with with_profile_time(feature_data_trans, whether_time_time_cudaevent=whether_time_time_cudaevent):
                x = x.to(device)
                y = y.to(device)
            with with_profile_time(train_time, whether_time_time_cudaevent=whether_time_time_cudaevent):
                y_hat = model(blocks, x)
                loss = F.cross_entropy(y_hat, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
        ## calculate the avg time of a list
        batch_prepare_time_avg = mean(batch_prepare_time[-80:-1])
        data_tansfer_time_avg = mean(data_tansfer_time[-80:-1])
        train_time_avg = mean(train_time[-80:-1])
        slice_time_avg = mean(slice_time[-80:-1])
        feature_data_transition_time_avg = mean(feature_data_trans[-80:-1])
        all_avg = (
            batch_prepare_time_avg
            + data_tansfer_time_avg
            + train_time_avg
            + slice_time_avg
            + feature_data_transition_time_avg
        )

        print("batch_prepare_time_avg: ", batch_prepare_time_avg)
        print("data_tansfer_time_avg: ", data_tansfer_time_avg)
        print("slice_time_avg: ", slice_time_avg)
        print("feature_data_transition_time_avg: ", feature_data_transition_time_avg)
        print("train_time_avg: ", train_time_avg)
        print("all_avg: ", all_avg)
        if epoch!=0:
            avg_epoch_batch_prepare_time.append(batch_prepare_time_avg)
            avg_epoch_data_tansfer_time.append(data_tansfer_time_avg)
            avg_epoch_slice_time.append(slice_time_avg)
            avg_epoch_feature_data_trans.append(feature_data_transition_time_avg)
            avg_epoch_train_time.append(train_time_avg)
            avg_epoch_all_time.append(all_avg)
        acc = evaluate(model, g, val_dataloader)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, total_loss / (it + 1), acc.item()
            )
        )
    print("avg_epoch_batch_prepare_time: ", mean(avg_epoch_batch_prepare_time))
    print("avg_epoch_data_tansfer_time: ", mean(avg_epoch_data_tansfer_time))
    print("avg_epoch_slice_time: ", mean(avg_epoch_slice_time))
    print("avg_epoch feature data trans", mean(avg_epoch_feature_data_trans))
    print("avg_epoch_train_time: ", mean(avg_epoch_train_time))
    print("avg_epoch_all_time: ", mean(avg_epoch_all_time))
    print('num of nodes: ', g.number_of_nodes())
    print('num of edges: ', g.number_of_edges())
    exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["cpu", "mixed", "puregpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    parser.add_argument(
        "--dataset", default='ogbn-products',choices=["ogbn-products", "ogbn-papers100M", "ogbn-arxiv",'ogbn-mag','ogbn-proteins','reddit','yelp']
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    ## 2 stand for using cuda event 3 stand for using time.time do not need to break down
    whether_time_time_cudaevent = 3
    print("whether_time_time_cudaevent: ", whether_time_time_cudaevent)

    import os

    filename = os.path.basename(__file__)
    print(filename)

    # load and preprocess dataset
    print("Loading data")
    print(args.mode)
    dataset_name = args.dataset
    # dataset = AsNodePredDataset(DglNodePropPredDataset(dataset_name))
    
    # import DglNodePropPredDataset
    # dataset = AsNodePredDataset(DglNodePropPredDataset(dataset_name))
    if dataset_name == "reddit":
        dataset =  AsNodePredDataset(dgl.data.RedditDataset())
    elif dataset_name == "yelp":
        dataset = AsNodePredDataset(dgl.data.YelpDataset())
    else:
        dataset = AsNodePredDataset(DglNodePropPredDataset(dataset_name))
    print("dataset: ", dataset_name)
    g = dataset[0]
    
    print('num of nodes: ', g.number_of_nodes())
    print('num of edges: ', g.number_of_edges())
    # g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")

    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    print("in_size", in_size)
    print("out_size", out_size)
    paper_100m = False
    if out_size == 172:
        paper_100m = True
    model = SAGE(in_size, 256, out_size).to(device)
    print("device: ", device)
    # print('model device: ', model.device)

    # model training
    print("Training...")

    proflie = True
    train(args, device, g, dataset, model)
