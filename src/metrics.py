import torch

def top_k_accuracy(output, target, top_k):
    # 上位 k 個の index を引っ張ってくる
    _, idx = output.topk(top_k)
    # 上位 k 個の index に target と一致しているものがあるか
    predicts = torch.Tensor([t in p for p, t in zip(idx, target)])
    # ミニバッチの acc を返す
    return predicts.sum() / len(predicts)
