import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# ✅ 统一引用
from src import path
from src.ChordGenerator_A import config

class MusicDataset(Dataset):
    def __init__(self, data_path):
        """
        读取 json 数据集 (train_data.json 或 val_data.json)
        """
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        取出第 idx 首歌，转换为 LongTensor
        返回: src (旋律), pos (位置), trg (和弦), length
        """
        item = self.data[idx]

        # 将 list 转为 tensor
        src = torch.LongTensor(item["input"])      # 旋律 Token IDs
        pos = torch.LongTensor(item["position"])   # [V3.1 新增] 绝对位置 IDs (0-31)
        trg = torch.LongTensor(item["target"])     # 和弦 Token IDs
        length = item["length"]                    # 真实长度

        return src, pos, trg, length


def collate_fn(batch):
    """
    【关键函数】Batch Padding
    """
    # 1. 拆包 (新增 pos_batch)
    src_batch, pos_batch, trg_batch, lengths = zip(*batch)

    # 2. 填充 (Padding)
    # src 和 trg 用 0 (vocab里的 <PAD>)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=config.PAD_IDX)
    trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=config.PAD_IDX)
    
    # [V3.1 新增] pos 用 POS_PAD_IDX (32) 填充
    # 因为 0 是有效的位置索引(第一拍)，不能用来做 Padding
    pos_padded = pad_sequence(pos_batch, batch_first=True, padding_value=config.POS_PAD_IDX)

    # 3. 转换为 Tensor
    lengths = torch.LongTensor(lengths)

    return src_padded, pos_padded, trg_padded, lengths


# --- 测试代码 ---
if __name__ == "__main__":
    # 测试加载
    import os
    if os.path.exists(path.TRAIN_DATASET_PATH):
        dataset = MusicDataset(str(path.TRAIN_DATASET_PATH))
        print(f"数据集大小: {len(dataset)}")

        # 模拟取一个样本
        src, pos, trg, l = dataset[0]
        print(f"Sample 0:")
        print(f" - Src shape: {src.shape}")
        print(f" - Pos shape: {pos.shape} (Example: {pos[:10].tolist()})")
        print(f" - Trg shape: {trg.shape}")
    else:
        print("⚠️ 未找到数据集文件，请先运行 tokenize_data.py")