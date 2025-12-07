import torch
import json
import os
import re

# 导入你的模型架构
try:
    from src.model import Encoder, Decoder, Seq2Seq
except ImportError:
    from model import Encoder, Decoder, Seq2Seq # 备用：方便你直接在 src 目录下测试运行

class AIComposer:
    def __init__(self, 
                 model_path=r'models\best_model.pth',     # 模型现在应该在这里
                 vocab_path=r'data\processed\vocab.json'): # 字典在这里
        """
        初始化 AI 作曲家
        1. 加载字典
        2. 重建模型结构
        3. 加载训练好的参数
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🤖 AI Composer 正在启动 (Device: {self.device})...")

        # 1. 加载字典
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"找不到字典文件: {vocab_path}")
            
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
            
        self.melody_stoi = self.vocab['melody'] # String to ID
        self.harmony_itos = {v: k for k, v in self.vocab['harmony'].items()} # ID to String (反向查找)
        
        # 2. 获取维度
        input_dim = len(self.melody_stoi)
        output_dim = len(self.vocab['harmony'])
        
        # 3. 初始化模型 (参数必须与 train.py 一致)
        # 如果你训练时改了参数，这里也要改
        ENC_EMB_DIM = 64
        DEC_EMB_DIM = 64
        HIDDEN_DIM = 128
        DROPOUT = 0.5
        
        enc = Encoder(input_dim, ENC_EMB_DIM, HIDDEN_DIM, DROPOUT)
        dec = Decoder(output_dim, DEC_EMB_DIM, HIDDEN_DIM, DROPOUT)
        
        self.model = Seq2Seq(enc, dec, self.device).to(self.device)
        
        # 4. 加载权重
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path} (请先运行 train.py)")
            
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval() # 切换到评估模式 (关闭 Dropout)
        print("✅ 模型加载完毕，准备作曲！")

    def clean_token(self, token):
        """(内部工具) 清洗旋律输入，逻辑与训练时保持一致"""
        token = str(token).strip()
        if token in ["_", "0"]: return token
        if token.isdigit(): return token
        return "0" # 脏数据转休止符

    def predict(self, melody_list):
        """
        【核心接口】给 B 组调用的函数
        Input:  ['60', '62', '_', '64'] (旋律列表)
        Output: ['C', 'C', 'C', 'G']    (和弦列表)
        """
        # 1. 预处理 (清洗 + 转数字)
        clean_seq = [self.clean_token(t) for t in melody_list]
        
        # 加上 <SOS> 和 <EOS>
        sos_id = self.melody_stoi["<SOS>"]
        eos_id = self.melody_stoi["<EOS>"]
        
        # 查表转换 (遇到没见过的音符就用休止符代替，防止报错)
        unk_id = self.melody_stoi.get("0", 3) 
        input_ids = [sos_id] + [self.melody_stoi.get(t, unk_id) for t in clean_seq] + [eos_id]
        
        # 转为 Tensor 并增加 Batch 维度 [1, seq_len]
        src_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
        src_len = torch.LongTensor([len(input_ids)])
        
        # 2. 推理 (Inference)
        with torch.no_grad(): # 不需要算梯度，节省内存
            # 编码器 (Encoder)
            encoder_outputs, hidden, cell = self.model.encoder(src_tensor, src_len)
            
            # 调整 hidden 格式适配 Decoder
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1).unsqueeze(0)
            cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1).unsqueeze(0)
            
            # 解码器 (Decoder) 逐步生成
            # 第一步输入 <SOS>
            trg_token = torch.tensor([self.vocab['harmony']["<SOS>"]], device=self.device)
            
            predicted_chords = []
            
            # 循环生成的长度 = 旋律的长度
            # 注意：我们要跳过 <SOS>，所以生成 len(clean_seq) 个
            for _ in range(len(clean_seq)):
                output, hidden, cell = self.model.decoder(trg_token, hidden, cell, encoder_outputs)
                
                # 取概率最大的那个 ID (Greedy Decode)
                top1 = output.argmax(1) 
                
                # 记录结果
                chord_str = self.harmony_itos[top1.item()]
                predicted_chords.append(chord_str)
                
                #把预测结果作为下一步的输入
                trg_token = top1
                
        return predicted_chords

# ================= 测试区域 =================
if __name__ == "__main__":
    # 模拟一段旋律 (Twinkle Twinkle Little Star: 1 1 5 5 6 6 5)
    # 对应 MIDI: 60 60 67 67 69 69 67
    test_melody = ['60', '60', '67', '67', '69', '69', '67', '_', '_', '0']
    
    try:
        composer = AIComposer()
        chords = composer.predict(test_melody)
        
        print("\n🎵 测试生成结果:")
        print(f"旋律: {test_melody}")
        print(f"和弦: {chords}")
        
        # 验证长度是否一致
        assert len(test_melody) == len(chords)
        print("✅ 长度对齐检查通过！")
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")