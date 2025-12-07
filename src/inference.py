import torch
import os
import sys

# ================= 动态导入处理 =================
# 兼容根目录运行(python interface.py) 和 src目录运行(python inference.py)
try:
    from src import config
    from src import utils
    from src.model import Encoder, Decoder, Seq2Seq
except ImportError:
    import config
    import utils
    from model import Encoder, Decoder, Seq2Seq

class AIComposer:
    def __init__(self):
        """
        初始化 AI 作曲家
        配置来源: src/config.py
        工具来源: src/utils.py
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🤖 AI Composer 正在初始化 (Device: {self.device})...")

        # 1. 加载字典
        print(f"📖 正在加载字典: {config.VOCAB_PATH}")
        self.vocab = utils.load_vocab(config.VOCAB_PATH)
        
        self.melody_stoi = self.vocab['melody']
        # 反向查找表: ID -> 和弦名
        self.harmony_itos = {v: k for k, v in self.vocab['harmony'].items()}
        
        # 2. 构建模型架构 (参数直接读取 config，确保与训练一致)
        input_dim = len(self.melody_stoi)
        output_dim = len(self.vocab['harmony'])
        
        enc = Encoder(input_dim, config.ENC_EMB_DIM, config.HIDDEN_DIM, config.DROPOUT)
        dec = Decoder(output_dim, config.DEC_EMB_DIM, config.HIDDEN_DIM, config.DROPOUT)
        
        self.model = Seq2Seq(enc, dec, self.device).to(self.device)
        
        # 3. 加载权重
        if not os.path.exists(config.MODEL_SAVE_PATH):
            raise FileNotFoundError(f"❌ 找不到模型文件: {config.MODEL_SAVE_PATH} (请先运行 train.py)")
            
        print(f"⚖️ 正在加载模型权重: {config.MODEL_SAVE_PATH}")
        # weights_only=True 解决 FutureWarning 安全警告
        self.model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=self.device, weights_only=True))
        
        self.model.eval() # 开启评估模式
        print("✅ AI Composer 准备就绪！")

    def predict(self, melody_list, temperature=1.0):
        """
        【核心接口】
        :param melody_list: 旋律 Token 列表
        :param temperature: 采样温度 (0.1=保守/死板, 1.0=正常, 1.2=狂野)
        """
        # 1. 清洗与转换 (调用 utils，保证一致性)
        clean_seq = [utils.clean_melody_token(t) for t in melody_list]
        src_tensor, src_len = utils.token_to_tensor(clean_seq, self.melody_stoi, self.device)
        
        predicted_chords = []
        
        with torch.no_grad():
            # A. 编码器 (Encoder)
            encoder_outputs, hidden, cell = self.model.encoder(src_tensor, src_len)
            
            # B. 调整 Hidden 状态
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1).unsqueeze(0)
            cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1).unsqueeze(0)
            
            # C. 解码器 (Decoder) 初始化输入 <SOS>
            sos_id = self.vocab['harmony'].get(config.SOS_TOKEN)
            trg_token = torch.tensor([sos_id], device=self.device)
            
            # 获取特殊符号 ID 用于后续逻辑控制
            hold_id = self.vocab['harmony'].get("_")
            pad_id = self.vocab['harmony'].get(config.PAD_TOKEN)

            # D. 循环生成
            for step in range(len(clean_seq)):
                output, hidden, cell = self.model.decoder(trg_token, hidden, cell, encoder_outputs)
                
                # ==========================================
                # 🌟 策略 1: 强制第一步不能是 "_" (HOLD)
                # ==========================================
                if step == 0 and hold_id is not None:
                    # 将 "_" 的 logits 设为负无穷，确保 softmax 后概率为 0
                    output[:, hold_id] = -float('inf')
                    if pad_id is not None:
                        output[:, pad_id] = -float('inf') # 也不能是 PAD

                # ==========================================
                # 🌟 策略 2: 温度采样 (Temperature Sampling)
                # ==========================================
                # 除以温度: 温度越低，差异被放大(越像 argmax)；温度越高，差异被缩小(越平均)
                logits = output / temperature
                
                # 转为概率分布
                probs = torch.softmax(logits, dim=1)
                
                # 按照概率进行随机抽样 (不再是死板的 argmax)
                top1 = torch.multinomial(probs, num_samples=1).squeeze(1)
                
                # 记录结果
                chord_str = self.harmony_itos.get(top1.item(), config.UNK_TOKEN)
                predicted_chords.append(chord_str)
                
                # 下一步的输入
                trg_token = top1
                
        return predicted_chords

# ================= 单元测试 =================
if __name__ == "__main__":
    try:
        composer = AIComposer()
        
        # 测试用例
        test_melody = ['60', '60', '67', '67', '69', '69', '67', '0']
        print(f"\n🎵 输入: {test_melody}")
        
        # 测试温度采样
        print("\n--- 温度测试 ---")
        print(f"Temp=0.8 (保守): {composer.predict(test_melody, temperature=0.8)}")
        print(f"Temp=1.2 (狂野): {composer.predict(test_melody, temperature=1.2)}")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")