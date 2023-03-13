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
import time
import numpy
import contextlib

sample_and_datatrans = []
data_tansfer_time = []
train_time = []
slice_time = []
start_time = 0
end_time = 0


def add_sample_datatransfer_time(end_time):
    sample_and_datatrans.append((end_time - start_time) * 1000)


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
    return MF.accuracy(
        torch.cat(y_hats), torch.cat(ys), task="multiclass", num_classes=out_size
    )

@contextlib.contextmanager
def with_profile_time(records, whether_time_time_cudaevent=2):
    if whether_time_time_cudaevent == 2:
        torch.cuda.synchronize()
        start = time.time()
        yield
        torch.cuda.synchronize()
        end = time.time()
        records.append((end - start) * 1000)
    elif whether_time_time_cudaevent == 3:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record(torch.cuda.current_stream())
        yield
        end_event.record(torch.cuda.current_stream())
        end_event.synchronize()
        step_elapsed = start_event.elapsed_time(end_event)
        records.append(step_elapsed)

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
    # create sampler & dataloader
    cpu_device = torch.device("cpu")
    # print('dataset device: ', dataset.train_idx.device)
    if uva_or_not == True:
        train_idx = dataset.train_idx.to(device)
        val_idx = dataset.val_idx.to(device)
    else:
        train_idx = dataset.train_idx.to(cpu_device)
        val_idx = dataset.val_idx.to(cpu_device)
    sampler = NeighborSampler(
        [15, 10, 5],  # fanout for [layer-0, layer-1, layer-2]
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    use_uva = args.mode == "mixed"
    if uva_or_not == False:
        use_uva = False
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=1024,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )
    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=1024,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    avg_epoch_batch_prepare_time = []
    avg_epoch_train_time = []
    avg_epoch_slice_time = []
    avg_epoch_all_time = []
    for epoch in range(5):
        model.train()
        total_loss = 0
        train_dataloader.sample_and_datatrans_time = []
        train_dataloader.slice_time = []
        train_dataloader.train_time = []
        train_dataloader.start_record_time = 0
        train_dataloader.end_record_time = 0
        train_dataloader.whether_time_time_cudaevent = whether_time_time_or_cudaevent
        val_dataloader.whether_time_time_cudaevent = 3
        torch.cuda.synchronize()
        if whether_time_time_or_cudaevent == 0:
            torch.cuda.synchronize()
            train_dataloader.start_record_time = time.time()
        else:
            train_dataloader.start_record_time = torch.cuda.Event(enable_timing=True)
            train_dataloader.end_record_time = torch.cuda.Event(enable_timing=True)
            train_dataloader.start_record_time.synchronize()
            train_dataloader.start_record_time.record(torch.cuda.current_stream())
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            print("it: ", it)
            if it > 100:
                break
            ## consider block device
            if train_dataloader.whether_time_time_cudaevent == 0:
                torch.cuda.synchronize()
                train_dataloader.end_record_time = time.time()
                train_dataloader.slice_time.append(
                    (
                        train_dataloader.end_record_time
                        - train_dataloader.start_record_time
                    )
                    * 1000
                )
                torch.cuda.synchronize()
                train_dataloader.start_record_time = time.time()
            else:
                train_dataloader.end_record_time.record(torch.cuda.current_stream())
                train_dataloader.end_record_time.synchronize()
                train_dataloader.slice_time.append(
                    train_dataloader.end_record_time.elapsed_time(
                        train_dataloader.start_record_time
                    )
                )
                train_dataloader.start_record_time.record(torch.cuda.current_stream())
            x = blocks[0].srcdata["feat"]
            if paper_100m == True:
                y = blocks[-1].dstdata["label"].to(torch.int64)
            else:
                y = blocks[-1].dstdata["label"]
            if train_dataloader.whether_time_time_cudaevent == 0:
                torch.cuda.synchronize()
                train_dataloader.end_record_time = time.time()

                train_dataloader.slice_time[-1] = (
                    train_dataloader.slice_time[-1]
                    + (
                        train_dataloader.end_record_time
                        - train_dataloader.start_record_time
                    )
                    * 1000
                )
                torch.cuda.synchronize()
                start_record_time = time.time()
            else:
                train_dataloader.end_record_time.record(torch.cuda.current_stream())
                train_dataloader.end_record_time.synchronize()

                train_dataloader.slice_time[-1] = train_dataloader.slice_time[
                    -1
                ] + train_dataloader.end_record_time.elapsed_time(
                    train_dataloader.start_record_time
                )
                torch.cuda.synchronize()
                train_dataloader.start_record_time.record(torch.cuda.current_stream())
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            if train_dataloader.whether_time_time_cudaevent == 0:
                torch.cuda.synchronize()
                end_record_time = time.time()
                train_dataloader.train_time.append(
                    (end_record_time - start_record_time) * 1000
                )
                torch.cuda.synchronize()
                train_dataloader.start_record_time = time.time()
            else:
                train_dataloader.end_record_time.record(torch.cuda.current_stream())
                train_dataloader.end_record_time.synchronize()
                train_dataloader.train_time.append(
                    train_dataloader.end_record_time.elapsed_time(
                        train_dataloader.start_record_time
                    )
                )
                torch.cuda.synchronize()
                train_dataloader.start_record_time.record(torch.cuda.current_stream())

        ## calculate the avg time of a list
        sample_and_datatrans_avg = mean(
            train_dataloader.sample_and_datatrans_time[-80:-1]
        )
        # data_tansfer_time_avg = mean(data_tansfer_time[-11:-1])
        train_time_avg = mean(train_dataloader.train_time[-80:-1])
        slice_time_avg = mean(train_dataloader.slice_time[-80:-1])
        all_avg = sample_and_datatrans_avg + train_time_avg + slice_time_avg

        print("sample_and_datatrans: ", train_dataloader.sample_and_datatrans_time)
        print("slice_time: ", train_dataloader.slice_time)
        print("train_time: ", train_dataloader.train_time)
        print("sample_and_datatrans_avg: ", sample_and_datatrans_avg)
        print("slice_time_avg: ", slice_time_avg)
        print("train_time_avg: ", train_time_avg)
        print("all_avg: ", all_avg)
        avg_epoch_batch_prepare_time.append(-1 * sample_and_datatrans_avg)
        avg_epoch_slice_time.append(-1 * slice_time_avg)
        avg_epoch_train_time.append(-1 * train_time_avg)
        avg_epoch_all_time.append(-1 * all_avg)
        # exit(0)
        acc = evaluate(model, g, val_dataloader)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, total_loss / (it + 1), acc.item()
            )
        )
    print("avg_epoch_batch_prepare_time: ", mean(avg_epoch_batch_prepare_time))
    print("avg_epoch_slice_time: ", mean(avg_epoch_slice_time))
    print("avg_epoch_train_time: ", mean(avg_epoch_train_time))
    print("avg_epoch_all_time: ", mean(avg_epoch_all_time))
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
        "--dataset", choices=["ogbn-products", "ogbn-papers100M", "ogbn-arxiv"]
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    import os

    filename = os.path.basename(__file__)
    print(filename)
    # load and preprocess dataset
    print("Loading data")
    print(args.mode)
    dataset_name = args.dataset

    whether_time_time_or_cudaevent = 0
    uva_or_not = True
    print("whether_time_time_or_cudaevent: ", whether_time_time_or_cudaevent)
    print("uva_or_not: ", uva_or_not)
    print("dataset_name: ", dataset_name)
    dataset = AsNodePredDataset(DglNodePropPredDataset(dataset_name))
    print("dataset: ", dataset_name)
    g = dataset[0]
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")

    # create GraphSAGE model
    # import pdb; pdb.set_trace()
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

    # test the model
    print("Testing...")
    acc = layerwise_infer(device, g, dataset.test_idx, model, batch_size=4096)
    print("Test Accuracy {:.4f}".format(acc.item()))
