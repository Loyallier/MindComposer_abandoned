import os
import json
import torch  # 假设您使用PyTorch进行Transformer训练

# ================= 配置区 =================
# 遵循 Melody_ 前缀规则
DATA_DIR = os.path.join('data', 'processed')
INPUT_FILE = os.path.join(DATA_DIR, 'dataset.txt')
VOCAB_FILE = os.path.join(DATA_DIR, 'Melody_vocab.json')

# =========================================

class MelodyTokenizer:
    def __init__(self):
        self.stoi = {} # String to Integer
        self.itos = {} # Integer to String
        self.vocab_size = 0

    def build_vocab(self, text_data):
        """
        根据文本数据构建词表
        """
        # 获取所有唯一字符并排序，保证确定性
        chars = sorted(list(set(text_data)))
        self.vocab_size = len(chars)
        
        # 构建映射
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        print(f"[Tokenizer] Vocab built. Size: {self.vocab_size}")
        print(f"[Tokenizer] Sample chars: {chars[:10]}")

    def save_vocab(self, filepath):
        """
        保存词表到JSON
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            # 注意：JSON的key必须是字符串，itos的key(int)在保存时会被转为str
            json.dump({'stoi': self.stoi, 'itos': self.itos}, f, indent=4)
        print(f"[Tokenizer] Vocab saved to {filepath}")

    def load_vocab(self, filepath):
        """
        从JSON加载词表
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocab file not found: {filepath}")
            
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.stoi = data['stoi']
            # 加载回来时，itos的key是str，需要转回int
            self.itos = {int(k): v for k, v in data['itos'].items()}
            self.vocab_size = len(self.stoi)
        
        print(f"[Tokenizer] Vocab loaded. Size: {self.vocab_size}")

    def encode(self, text):
        """
        将字符串转换为整数列表
        """
        return [self.stoi[c] for c in text]

    def decode(self, indices):
        """
        将整数列表（或Tensor）转换为字符串
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return ''.join([self.itos[i] for i in indices])

def main():
    # 1. 读取清洗后的数据
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Please run Melody_preprocess.py first.")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        text_data = f.read()

    # 2. 初始化并构建 Tokenizer
    tokenizer = MelodyTokenizer()
    tokenizer.build_vocab(text_data)

    # 3. 保存词表
    tokenizer.save_vocab(VOCAB_FILE)

    # 4. 验证编码/解码逻辑 (Sanity Check)
    print("-" * 30)
    print("Sanity Check: Encoding/Decoding")
    
    # 取前50个字符进行测试
    sample_text = text_data[:50]
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {repr(sample_text)}")
    print(f"Encoded : {encoded}")
    print(f"Decoded : {repr(decoded)}")
    
    # 逻辑验证：必须完全还原
    if sample_text == decoded:
        print(">> TEST PASSED: Perfect reconstruction.")
    else:
        print(">> TEST FAILED: Data mismatch.")
    print("-" * 30)

    # 5. 演示 Tensor 转换
    tensor_data = torch.tensor(encoded, dtype=torch.long)
    print(f"Tensor Shape: {tensor_data.shape}")
    print(f"Tensor Type:  {tensor_data.dtype}")

if __name__ == "__main__":
    main()