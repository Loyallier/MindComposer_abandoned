import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import pickle # 用于可能的缓存，本例暂不使用但保留扩展性
from Melody_tokenizer import MelodyTokenizer # 确保文件名大小写匹配

class MelodyDataset(Dataset):
    def __init__(self, data_path, tokenizer, block_size=128):
        """
        初始化数据集
        :param data_path: 预处理后的dataset.txt路径
        :param tokenizer: 已构建并加载词表的Tokenizer实例
        :param block_size: 上下文窗口大小 (Context Length)
        """
        self.tokenizer = tokenizer
        self.block_size = block_size

        # 1. 读取数据
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        # 2. Tokenization (全量数据转Tensor)
        # 对于1000首歌，数据量极小，直接全部加载进内存是最高效的
        print(f"[Dataset] Tokenizing {len(raw_text)} characters...")
        self.data_ids = torch.tensor(self.tokenizer.encode(raw_text), dtype=torch.long)
        print(f"[Dataset] Tokenization complete. Total tokens: {len(self.data_ids)}")

    def __len__(self):
        # 数据集长度 = 总token数 - block_size
        # 这样确保最后一个样本也有完整的 block_size + 1 (用于target)
        return len(self.data_ids) - self.block_size

    def __getitem__(self, idx):
        # 逻辑：
        # 我们需要提取长度为 block_size + 1 的片段
        # chunk = [token_i, token_i+1, ..., token_i+block_size]
        # x (输入) = chunk[:-1] (不含最后一个)
        # y (目标) = chunk[1:]  (不含第一个，即向后位移一位)
        
        chunk = self.data_ids[idx : idx + self.block_size + 1]
        
        x = chunk[:-1]
        y = chunk[1:]
        
        return x, y

def create_dataloaders(data_path, vocab_path, block_size=128, batch_size=64, train_split=0.9):
    """
    工厂函数：创建训练集和验证集的DataLoader
    """
    # 1. 准备Tokenizer
    tokenizer = MelodyTokenizer()
    if os.path.exists(vocab_path):
        tokenizer.load_vocab(vocab_path)
    else:
        # 如果没有现成的vocab，逻辑上应该报错或临时构建，
        # 这里为了严谨，要求必须先运行 Melody_tokenizer.py 生成 vocab
        raise FileNotFoundError(f"Vocab file not found at {vocab_path}. Run Melody_tokenizer.py first.")

    # 2. 实例化Dataset
    full_dataset = MelodyDataset(data_path, tokenizer, block_size)

    # 3. 划分训练/验证集
    # 对于过拟合实验，通常可以使用全部数据训练。
    # 但为了逻辑严谨性，我们保留标准的划分接口。
    # 如果您想纯粹过拟合，可以将 train_split 设为 1.0 (需处理余数报错，建议 0.99)
    train_size = int(len(full_dataset) * train_split)
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 4. 创建DataLoader
    # pin_memory=True 对于GPU训练能加速数据传输
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0) # Windows下建议num_workers=0避免多进程报错
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    print(f"[DataLoader] Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")
    print(f"[DataLoader] Batch size: {batch_size} | Block size: {block_size}")
    
    return train_loader, val_loader, tokenizer

# ================= 测试代码 =================
if __name__ == "__main__":
    # 模拟路径
    PROCESSED_DATA = os.path.join('data', 'processed', 'dataset.txt')
    VOCAB_FILE = os.path.join('data', 'processed', 'Melody_vocab.json')

    # 测试参数
    BATCH_SIZE = 4 # 测试用小Batch
    BLOCK_SIZE = 10 # 测试用小窗口

    try:
        train_loader, val_loader, tokenizer = create_dataloaders(
            PROCESSED_DATA, 
            VOCAB_FILE, 
            block_size=BLOCK_SIZE, 
            batch_size=BATCH_SIZE
        )

        # 获取一个Batch查看数据形态
        x_batch, y_batch = next(iter(train_loader))
        
        print("-" * 30)
        print("Data Shape Check:")
        print(f"Input (x) shape: {x_batch.shape}") # 预期: [4, 10]
        print(f"Target (y) shape: {y_batch.shape}")# 预期: [4, 10]
        
        print("\nLogic Check (First sample in batch):")
        print(f"x[0]: {x_batch[0].tolist()}")
        print(f"y[0]: {y_batch[0].tolist()}")
        
        decoded_x = tokenizer.decode(x_batch[0])
        decoded_y = tokenizer.decode(y_batch[0])
        
        print(f"\nText x: {repr(decoded_x)}")
        print(f"Text y: {repr(decoded_y)}")
        
        # 逻辑验证：y的第一个字符应该是x的第二个字符
        if decoded_x[1:] == decoded_y[:-1]:
            print(">> LOGIC VERIFIED: Target is correctly shifted.")
        else:
            print(">> LOGIC ERROR: Target mismatch.")
        print("-" * 30)

    except Exception as e:
        print(f"Test failed: {e}")