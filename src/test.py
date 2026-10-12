import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
from dataset import BronchoDataset, collate_moe
from torch.utils.data import DataLoader
import numpy as np
import random
import argparse
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
import pandas as pd

from train import define_model, test_one_epoch

def find_best_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_df_path", type=str, default="./test_data.csv")
    parser.add_argument("--ct_root", type=str, default="./patches")
    parser.add_argument("--graph_root", type=str, default="./graphs")
    parser.add_argument("--lung_root", type=str, default="./lung_masks")
    parser.add_argument("--lobe_root", type=str, default="./lobe_masks")
    parser.add_argument("--size", type=tuple, default=(128, 128, 32))
    parser.add_argument("--save_dir", type=str, default="./results/")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    test_dataset = BronchoDataset(args.test_df_path, args.ct_root, args.graph_root, args.lung_root, args.lobe_root, args.size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, collate_fn=collate_moe)
    model = define_model()
    model.load_state_dict(torch.load(f"{args.save_dir}/best_model.pth"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_acc, test_f1, test_roc_auc, test_pr_auc, test_weights, fnames, y_true, y_pred, y_prob = test_one_epoch(model, test_loader, device)
    print(f"Test Accuracy: {test_acc}, Test F1: {test_f1}, Test ROC AUC: {test_roc_auc}, Test PR AUC: {test_pr_auc}, Test Weights: {test_weights}")
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_prob):
        y_prob = y_prob.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    best_thresh = find_best_threshold(y_true, y_prob)
    y_pred = (y_prob >= best_thresh).astype(int)
    print(f"Best Threshold: {best_thresh}")
    acc, f1, roc_auc, pr_auc = accuracy_score(y_true, y_pred), f1_score(y_true, y_pred), test_roc_auc, test_pr_auc
    print(f"Test Accuracy: {acc}, Test F1: {f1}, Test ROC AUC: {roc_auc}, Test PR AUC: {pr_auc}")
    # save the metrics and results to a csv file
    metrics = pd.DataFrame([{
        "test_acc": acc,
        "test_f1": f1,
        "test_roc_auc": roc_auc,
        "test_pr_auc": pr_auc,
    }])
    metrics.to_csv(f"{args.save_dir}/test_metrics_all.csv", index=False)
    # save the results to a csv file
    fnames = [fname[0] for fname in fnames]
    y_true = [y_true for y_true in y_true]
    y_pred = [y_pred for y_pred in y_pred]
    y_prob = [y_prob for y_prob in y_prob]
    results = pd.DataFrame({
        "fname": fnames,
        "label": y_true,
        "pred": y_pred,
        "prob": y_prob,
        "best_thresh": best_thresh,
    })
    results.to_csv(f"{args.save_dir}/test_results_all.csv", index=False)

if __name__ == "__main__":
    main()