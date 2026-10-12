import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch.nn.functional as F
import torch
from dataset import BronchoDataset, collate_moe
from model import AaMoEModel, GlobalCTExpert, LobeExpert, PathSequenceExpert
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
import random
import argparse
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc

def define_model():
    experts = [
        GlobalCTExpert(),
        LobeExpert(),
        PathSequenceExpert(in_dim=13),
    ]
    model = AaMoEModel(experts)
    return model

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for batch in loader:
        # batch 是 MoEBatch
        if batch.ct is not None:
            batch.ct = batch.ct.to(device)
        batch.graph = batch.graph.to(device)
        batch.target_pos_mm = batch.target_pos_mm.to(device)
        if batch.target_pos_norm is not None:
            batch.target_pos_norm = batch.target_pos_norm.to(device)
        batch.lobe_id = batch.lobe_id.to(device)
        batch.label = batch.label.to(device)
        for key, value in batch.extra.items():
            batch.extra[key] = value.to(device)
        out = model(batch)
        logit = out["logit"]
        y = batch.label

        loss = F.binary_cross_entropy_with_logits(logit, y)

        entropy_loss = out["entropy_loss"]
        loss = loss + 0.01 * entropy_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += float(loss.item())
    return total / max(1, len(loader))

def test_one_epoch(model, loader, device):
    model.eval()
    logits = []
    labels = []
    preds = []
    weights = []
    fnames = []
    probs = []
    for batch in loader:
        batch.ct = batch.ct.to(device)
        batch.graph = batch.graph.to(device)
        batch.target_pos_mm = batch.target_pos_mm.to(device)
        batch.target_pos_norm = batch.target_pos_norm.to(device)
        batch.lobe_id = batch.lobe_id.to(device)
        batch.label = batch.label.to(device)
        batch.fname = batch.fname
        # batch.extra = batch.extra.to(device)
        for key, value in batch.extra.items():
            batch.extra[key] = value.to(device)
        out = model(batch)
        logit = out["logit"]
        y = batch.label
        prob = logit
        pred = torch.round(torch.sigmoid(logit))
        probs.append(prob.detach().cpu())
        logits.append(logit.detach().cpu())
        labels.append(y.detach().cpu())
        preds.append(pred.detach().cpu())
        weights.append(out["gate_w"].detach().cpu())
        fnames.append(batch.fname)
    # logits = torch.cat(logits, dim=0)
    probs = torch.cat(probs, dim=0)
    labels = torch.cat(labels, dim=0)
    preds = torch.cat(preds, dim=0)
    weights = torch.cat(weights, dim=0)
    # compute accuracy, f1, roc_auc, pr_auc
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    roc_auc = roc_auc_score(labels, probs)
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    weights = weights.mean(dim=0)
    return accuracy, f1, roc_auc, pr_auc, weights, fnames, labels, preds, probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_df_path", type=str, default="./train_data.csv")
    parser.add_argument("--test_df_path", type=str, default="./test_data.csv")
    parser.add_argument("--ct_root", type=str, default="./patches")
    parser.add_argument("--graph_root", type=str, default="./graphs")
    parser.add_argument("--lung_root", type=str, default="./lung_masks")
    parser.add_argument("--lobe_root", type=str, default="./lobe_masks")
    parser.add_argument("--save_dir", type=str, default="./results/")
    parser.add_argument("--size", type=tuple, default=(128, 128, 32))
    args = parser.parse_args()
    
    train_dataset = BronchoDataset(args.train_df_path, args.ct_root, args.graph_root, args.lung_root, args.lobe_root, args.size)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, collate_fn=collate_moe)
    test_dataset = BronchoDataset(args.test_df_path, args.ct_root, args.graph_root, args.lung_root, args.lobe_root, args.size)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8, collate_fn=collate_moe)
    model = define_model()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # writer = SummaryWriter(log_dir=f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(args.save_dir, exist_ok=True)
    best_test_roc_auc = 0
    for epoch in range(30):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch}, Train Loss: {train_loss}")
        test_acc, test_f1, test_roc_auc, test_pr_auc, test_weights, _, _, _, _ = test_one_epoch(model, test_loader, device)
        print(f"Epoch {epoch}, Test Accuracy: {test_acc}, Test F1: {test_f1}, Test ROC AUC: {test_roc_auc}, Test PR AUC: {test_pr_auc}, Test Weights: {test_weights}")
        if test_roc_auc > best_test_roc_auc:
            best_test_roc_auc = test_roc_auc
            torch.save(model.state_dict(), f'{args.save_dir}/best_model_{best_test_roc_auc:4f}.pth')
        # writer.add_scalar("train_loss", train_loss, epoch)
    # writer.close()

if __name__ == "__main__":
    main()