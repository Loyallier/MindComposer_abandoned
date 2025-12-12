import torch
import os
import sys

# 1. 路径挂载逻辑 (保持不变)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"🔧 已将根目录挂载: {project_root}")

from src import path
from src.ChordGenerator_A import config
from src.ChordGenerator_A import utils
from src.ChordGenerator_A.model import Encoder, Decoder, Seq2Seq

class AIComposer:
    def __init__(self):
        """
        初始化 AI 作曲家
        配置来源: src/config.py
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🤖 AI Composer 正在初始化 (Device: {self.device})...")

        # 1. 加载字典
        if not os.path.exists(config.VOCAB_PATH):
             raise FileNotFoundError(f"❌ 找不到字典文件: {config.VOCAB_PATH}")
             
        self.vocab = utils.load_vocab(config.VOCAB_PATH)
        self.melody_stoi = self.vocab['melody']
        self.harmony_itos = {v: k for k, v in self.vocab['harmony'].items()}
        
        # 2. 构建模型
        input_dim = len(self.melody_stoi)
        output_dim = len(self.vocab['harmony'])
        
        enc = Encoder(input_dim, config.ENC_EMB_DIM, config.HIDDEN_DIM, config.DROPOUT)
        dec = Decoder(output_dim, config.DEC_EMB_DIM, config.HIDDEN_DIM, config.DROPOUT)
        
        self.model = Seq2Seq(enc, dec, self.device).to(self.device)
        
        # 3. 加载权重
        if not os.path.exists(config.MODEL_SAVE_PATH):
            raise FileNotFoundError(f"❌ 找不到模型权重: {config.MODEL_SAVE_PATH}")
            
        print(f"⚖️ 正在加载模型权重: {config.MODEL_SAVE_PATH}")
        self.model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=self.device, weights_only=True))
        self.model.eval()
        print("✅ AI Composer 准备就绪！")

    def predict(self, melody_list):
        """
        :param melody_list: 旋律 Token 列表 (e.g. ['60', '_', '<BAR>', '62'])
        :return: 和弦列表 (不含 BAR)
        """
        # 参数从 config 读取
        temperature = config.INFERENCE_TEMP
        top_k = config.INFERENCE_TOP_K
        
        # 1. 清洗与转换
        clean_seq = [utils.clean_melody_token(t) for t in melody_list]
        src_tensor, src_len = utils.token_to_tensor(clean_seq, self.melody_stoi, self.device)
        
        predicted_chords = []
        
        with torch.no_grad():
            # A. Encoder
            encoder_outputs, hidden, cell = self.model.encoder(src_tensor, src_len)
            
            # B. Bridge Hidden State
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1).unsqueeze(0)
            cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1).unsqueeze(0)
            
            # C. Decoder Input (<SOS>)
            sos_id = self.vocab['harmony'].get(config.SOS_TOKEN)
            trg_token = torch.tensor([sos_id], device=self.device)
            
            # D. Step-by-Step Generation
            for step, input_token_str in enumerate(clean_seq):
                # 运行一步 Decoder
                output, hidden, cell = self.model.decoder(trg_token, hidden, cell, encoder_outputs)
                
                # 🌟 特殊逻辑：如果是 <BAR>，仅运行模型更新状态，不采样输出
                if input_token_str == config.BAR_TOKEN:
                    # 我们假设当旋律是 BAR 时，对应的和弦目标也是 BAR (训练时是这样对齐的)
                    # 所以我们可以强制把下一个输入设为 BAR，或者让模型自己跑
                    # 这里为了稳定性，我们强制让下一个时刻的输入也是 BAR
                    bar_id = self.vocab['harmony'].get(config.BAR_TOKEN)
                    trg_token = torch.tensor([bar_id], device=self.device)
                    continue 

                # --- 正常采样逻辑 ---
                logits = output / temperature
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=1)
                top_k_probs = torch.softmax(top_k_logits, dim=1)
                sample_idx = torch.multinomial(top_k_probs, num_samples=1).squeeze(1)
                top1 = top_k_indices.gather(1, sample_idx.unsqueeze(1)).squeeze(1)
                
                # 记录结果
                chord_str = self.harmony_itos.get(top1.item(), config.UNK_TOKEN)
                predicted_chords.append(chord_str)
                
                # 更新下一个输入
                trg_token = top1
                
        return predicted_chords

if __name__ == "__main__":
    composer = AIComposer()
    # 测试带 BAR 的序列
    # 期望：输出列表里不包含任何 BAR 符号，且长度比输入少 1 (因为 BAR 被吞了)
    test_melody = ['60', '_', '_', '_', config.BAR_TOKEN, '62', '_', '_', '_']
    print(f"\n🎵 输入: {test_melody}")
    result = composer.predict(test_melody)
    print(f"🎹 输出: {result}")
    print(f"📏 长度验证: 输入={len(test_melody)}, 输出={len(result)} (应相差1)")