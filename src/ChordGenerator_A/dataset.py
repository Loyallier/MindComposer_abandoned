import json
import torch
import random
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# ✅ 统一引用
from src import path
from src.ChordGenerator_A import config

# ✅ 统一引用
from src import path
from src.ChordGenerator_A import config

class MusicDataset(Dataset):
    def __init__(self, data_path, augment=False):
        """
        [V3.3 更新] 新增 augment 参数
        augment=True: 开启数据增强（加噪），用于训练集
        augment=False: 保持原样，用于验证集
        """
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.augment = augment  # 保存标记

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 获取原始 list (注意：这是一个引用，修改它会影响内存中的数据)
        src_list = item["input"]
        
        # 🛡️ [V3.3 核心] 动态旋律加噪 (Input Noise)
        # 仅在 augment=True 时触发
        if self.augment:
            # 必须 copy，否则会永久修改内存中的 dataset，导致下一轮 epoch 数据越来越乱
            src_list = src_list.copy() 
            
            # 避开开头的 SOS 和结尾的 EOS
            # 假设 list 结构是 [SOS, n1, n2, ..., EOS, 31]
            # 我们只破坏中间的音符
            seq_len = len(src_list)
            if seq_len > 2:
                for i in range(1, seq_len - 1):
                    # 跳过特殊符号 (如 BAR=config.BAR_TOKEN 或其 ID)
                    # 这里 src_list 里存的是 ID 还是 Token? 
                    # 根据 tokenize_data.py，json 里存的是 ID (int)
                    # 假设 config.BAR_TOKEN 对应的 ID 不应该被 mask
                    # 📉 [V3.5] 降低难度: 0.15 -> 0.10 (10% 噪声)
                    if random.random() < 0.10:
                        # 将其替换为 "0" (UNK/PAD 的 ID，通常是 0)
                        # 强迫模型去"猜"这个音
                        src_list[i] = config.PAD_IDX

        # 转为 Tensor
        src = torch.LongTensor(src_list)
        pos = torch.LongTensor(item["position"])
        trg = torch.LongTensor(item["target"])
        length = item["length"]

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