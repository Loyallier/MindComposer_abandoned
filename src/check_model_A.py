# check_model.py
import torch
from model import Encoder, Decoder, Seq2Seq
from dataset import MusicDataset, collate_fn
from torch.utils.data import DataLoader

# 1. 加载一小部分数据
dataset = MusicDataset("data\\processed\\dataset_encoded.json")
loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
src, trg, lengths = next(iter(loader))

print(f"输入形状: {src.shape}") # Expect: [2, max_len]
print(f"目标形状: {trg.shape}") # Expect: [2, max_len]

# 2. 初始化模型
INPUT_DIM = 37 # 假设 melody 词表大小
OUTPUT_DIM = 32 # 假设 harmony 词表大小
ENC_EMB_DIM = 64  # 稍微加大一点，让它看得更细
DEC_EMB_DIM = 64
HIDDEN_DIM = 128  # 128 是处理音乐序列的标准配置
DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HIDDEN_DIM, DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HIDDEN_DIM, DROPOUT)
model = Seq2Seq(enc, dec, device='cpu')

# 3. 试运行一次 (Forward Pass)
# 只要这一步不报错，A 组的任务就完成了一大半
outputs = model(src, trg, lengths)
print(f"模型输出形状: {outputs.shape}") # Expect: [2, max_len, OUTPUT_DIM]
print("✅ 模型构建成功！可以开始写 train.py 了。")