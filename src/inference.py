import torch
import os
import sys

# ================= 动态导入处理 =================
# 这样写是为了兼容两种运行方式：
# 1. 在根目录运行 python interface.py (作为模块被引用)
# 2. 在 src 目录运行 python inference.py (单独测试)
try:
    from src import config
    from src import utils
    from src.model import Encoder, Decoder, Seq2Seq
except ImportError:
    # 如果找不到 src，说明可能是在 src 目录下直接运行的
    import config
    import utils
    from model import Encoder, Decoder, Seq2Seq

class AIComposer:
    def __init__(self):
        """
        初始化 AI 作曲家
        所有路径和参数都直接从 config 读取，确保与训练时完全一致。
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🤖 AI Composer 正在初始化 (Device: {self.device})...")

        # 1. 加载字典 (调用 utils 通用函数)
        # config.VOCAB_PATH 统一管理路径
        print(f"📖 正在加载字典: {config.VOCAB_PATH}")
        self.vocab = utils.load_vocab(config.VOCAB_PATH)
        
        self.melody_stoi = self.vocab['melody']
        # 创建反向查找表: ID -> 和弦名
        self.harmony_itos = {v: k for k, v in self.vocab['harmony'].items()}
        
        # 2. 构建模型架构
        # 参数直接从 config 读取，避免手写出错
        input_dim = len(self.melody_stoi)
        output_dim = len(self.vocab['harmony'])
        
        enc = Encoder(input_dim, config.ENC_EMB_DIM, config.HIDDEN_DIM, config.DROPOUT)
        dec = Decoder(output_dim, config.DEC_EMB_DIM, config.HIDDEN_DIM, config.DROPOUT)
        
        self.model = Seq2Seq(enc, dec, self.device).to(self.device)
        
        # 3. 加载训练好的权重
        if not os.path.exists(config.MODEL_SAVE_PATH):
            raise FileNotFoundError(f"❌ 找不到模型文件: {config.MODEL_SAVE_PATH} (请检查 config.py 路径或先运行 train.py)")
            
        print(f"⚖️ 正在加载模型权重: {config.MODEL_SAVE_PATH}")
        self.model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=self.device))
        
        # 切换到评估模式 (关闭 Dropout 等训练专用功能)
        self.model.eval()
        print("✅ AI Composer 准备就绪！")

    def predict(self, melody_list):
        """
        【核心接口】
        Input:  ['60', '62', '_', '64']
        Output: ['C', 'Dm', 'G7', 'C']
        """
        # 👇【新增 Debug】看看传入了什么
        print(f"👀 [Debug] 收到旋律输入: {melody_list}")

        # 1. 清洗数据 (调用 utils，确保与训练逻辑一致)
        # 哪怕 B 组传进来脏数据，这里也会自动清洗
        clean_seq = [utils.clean_melody_token(t) for t in melody_list]
        
        # 👇【新增 Debug】看看清洗后剩下了什么
        print(f"🧼 [Debug] 清洗后数据: {clean_seq}")

        # 2. 转换为 Tensor (调用 utils)
        # 自动加上 SOS, EOS 并处理 device
        src_tensor, src_len = utils.token_to_tensor(clean_seq, self.melody_stoi, self.device)
        
        # 3. 模型推理 (Inference)
        predicted_chords = []
        
        with torch.no_grad(): # 这一步不需要算梯度，省显存
            # A. 编码器 (Encoder) 理解旋律
            encoder_outputs, hidden, cell = self.model.encoder(src_tensor, src_len)
            
            # B. 调整 hidden 状态以适配 Decoder (拼接双向 LSTM 的状态)
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1).unsqueeze(0)
            cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1).unsqueeze(0)
            
            # C. 解码器 (Decoder) 逐个生成和弦
            # 起始信号 <SOS>
            trg_token = torch.tensor([self.vocab['harmony'][config.SOS_TOKEN]], device=self.device)
            
            # 👇【新增 Debug】看看循环了多少次
            print(f"🔄 [Debug] 准备生成 {len(clean_seq)} 个和弦...")

            # 循环次数 = 旋律长度
            for _ in range(len(clean_seq)):
                output, hidden, cell = self.model.decoder(trg_token, hidden, cell, encoder_outputs)
                
                # 取概率最大的那个 ID (Greedy Decode)
                top1 = output.argmax(1)
                
                # 将 ID 转回字符串
                chord_str = self.harmony_itos.get(top1.item(), config.UNK_TOKEN)
                predicted_chords.append(chord_str)
                
                # 当前的预测结果不仅是输出，也是下一步的输入
                trg_token = top1
                
        # 👇【新增 Debug】看看最后生成了什么
        print(f"🎹 [Debug] 最终生成和弦: {predicted_chords}")
        return predicted_chords

# ================= 单元测试 =================
if __name__ == "__main__":
    # 这里是用来测试 inference.py 本身能不能跑通的
    # 不会被 interface.py 调用
    try:
        composer = AIComposer()
        
        # 测试用例: Twinkle Twinkle Little Star
        test_melody = ['60', '60', '67', '67', '69', '69', '67', '0']
        print(f"\n🎵 输入旋律: {test_melody}")
        
        result = composer.predict(test_melody)
        print(f"🎹 生成和弦: {result}")
        
        if len(result) == len(test_melody):
            print("✅ 测试通过：长度对齐！")
        else:
            print("⚠️ 测试警告：长度不对齐！")
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        print("提示：请检查 config.py 里的路径是否正确，或者是否已经运行过 train.py 生成了模型。")