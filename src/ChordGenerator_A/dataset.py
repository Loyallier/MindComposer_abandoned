# import json
# import torch
# from torch.utils.data import Dataset
# from torch.nn.utils.rnn import pad_sequence

# # Path import
# try:
#     from src import path
# except ImportError:
# import path

import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from src import path

class MusicDataset(Dataset):
    def __init__(self, data_path):
        """
        读取 dataset_encoded.json
        """
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        取出第 idx 首歌，转换为 LongTensor
        """
        item = self.data[idx]

        # 将 list 转为 tensor
        src = torch.LongTensor(item["input"])  # 旋律 (Encoder 输入)
        trg = torch.LongTensor(item["target"])  # 和弦 (Decoder 目标)
        length = item["length"]  # 真实长度

        return src, trg, length


def collate_fn(batch):
    """
    【关键函数】
    DataLoader 取出一个 Batch (比如32首歌) 时，
    因为每首长度不一样，这里负责把它们补齐 (Pad) 到同样长度。
    """
    # 1. 拆包
    src_batch, trg_batch, lengths = zip(*batch)

    # 2. 填充 (Padding)
    # batch_first=True -> 输出形状 (Batch, Seq_Len)
    # padding_value=0 -> 用 0 (vocab里的 <PAD>) 来填充
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=0)

    # 3. 转换为 Tensor
    lengths = torch.LongTensor(lengths)

    return src_padded, trg_padded, lengths


# --- 测试代码 (你可以直接运行这个文件检查数据加载是否正常) ---
if __name__ == "__main__":
    # 假设你已经有了 'dataset_encoded.json'
    dataset = MusicDataset(str(path.DATASET_PATH))
    print(f"数据集大小: {len(dataset)}")

    # 模拟取一个样本
    src, trg, l = dataset[0]
    print(f"第一条数据 Input shape: {src.shape}, Target shape: {trg.shape}")
