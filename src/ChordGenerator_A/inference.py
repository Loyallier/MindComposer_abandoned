# import torch
# import os

# from src import path

# # ================= 动态导入处理 =================
# # 兼容根目录运行(python interface.py) 和 src目录运行(python inference.py)
# try:
#     from src.ChordGenerator_A import config
#     from src.ChordGenerator_A import utils
#     from src.ChordGenerator_A.model import Encoder, Decoder, Seq2Seq
# except ImportError:
#     # 如果找不到 src，说明可能是在 src 目录下直接运行的
#     import src.ChordGenerator_A.config as config
#     import src.ChordGenerator_A.utils as utils
#     from src.ChordGenerator_A.model import Encoder, Decoder, Seq2Seq

import torch
import os
import config # 导入 src/ChordGenerator_A/config.py
import utils  # 导入 src/ChordGenerator_A/utils.py
from model import Encoder, Decoder, Seq2Seq
from src import path

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
        if not os.path.exists(config.VOCAB_PATH):
             raise FileNotFoundError(f"❌ 找不到字典文件: {config.VOCAB_PATH} (请检查 src/config.py 路径)")
             
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

    def predict(self, melody_list, temperature=1.0, top_k=3):
        """
        【核心接口】
        :param melody_list: 旋律 Token 列表
        :param temperature: 温度 (建议 0.8-1.0)。越低越保守。
        :param top_k: 只在概率最高的 k 个选项里抽样 (防止乱猜)。
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
            
            # D. 循环生成
            for step in range(len(clean_seq)):
                output, hidden, cell = self.model.decoder(trg_token, hidden, cell, encoder_outputs)
                
                # ==========================================
                # 🌟 Top-K 采样逻辑 (解决 AI 偷懒的关键)
                # ==========================================
                
                # 应用温度
                logits = output / temperature
                
                # 只保留前 K 个最大的概率
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=1)
                
                # 对这 K 个进行 softmax
                top_k_probs = torch.softmax(top_k_logits, dim=1)
                
                # 在这 K 个里抽签
                sample_idx_in_top_k = torch.multinomial(top_k_probs, num_samples=1).squeeze(1)
                
                # 映射回原始词表的 ID
                top1 = top_k_indices.gather(1, sample_idx_in_top_k.unsqueeze(1)).squeeze(1)

                # ==========================================

                chord_str = self.harmony_itos.get(top1.item(), config.UNK_TOKEN)
                predicted_chords.append(chord_str)
                trg_token = top1
                
        return predicted_chords

# ================= 单元测试 =================
if __name__ == "__main__":
    try:
        print("正在测试 inference.py ...")
        composer = AIComposer()
        
        # 测试用例
        test_melody = ['60', '60', '67', '67', '69', '69', '67', '0']
        print(f"\n🎵 输入: {test_melody}")
        
        # 对比测试：Top-K=1 (死板) vs Top-K=3 (灵活)
        print(f"🎹 输出 (Top-K=1): {composer.predict(test_melody, top_k=1)}")
        print(f"🎹 输出 (Top-K=3): {composer.predict(test_melody, top_k=3)}")
        print("✅ 测试通过！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")