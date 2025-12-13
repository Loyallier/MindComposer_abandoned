import torch
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import path
from src.ChordGenerator_A import config
from src.ChordGenerator_A import utils
from src.ChordGenerator_A.model import Encoder, Decoder, Seq2Seq

class AIComposer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # ... (Vocab loading logic same as before) ...
        self.vocab = utils.load_vocab(config.VOCAB_PATH)
        self.melody_stoi = self.vocab['melody']
        self.harmony_itos = {v: k for k, v in self.vocab['harmony'].items()}
        
        # 模型初始化
        input_dim = len(self.melody_stoi)
        output_dim = len(self.vocab['harmony'])
        
        # ✅ 必须传入 pos 参数
        enc = Encoder(input_dim, config.ENC_EMB_DIM, config.HIDDEN_DIM, config.DROPOUT, 
                      config.POS_VOCAB_SIZE, config.POS_EMB_DIM)
        dec = Decoder(output_dim, config.DEC_EMB_DIM, config.HIDDEN_DIM, config.DROPOUT)
        
        self.model = Seq2Seq(enc, dec, self.device).to(self.device)
        
        if not os.path.exists(config.MODEL_SAVE_PATH):
            raise FileNotFoundError(f"❌ Weights not found: {config.MODEL_SAVE_PATH}")
            
        self.model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=self.device))
        self.model.eval()

    def predict(self, melody_list, pos_list):
        """
        [V3.1 Update]
        :param melody_list: ['60', '62', '<BAR>']
        :param pos_list:    [14,   15,    31] (外部计算好的绝对位置)
        """
        temperature = config.INFERENCE_TEMP
        top_k = config.INFERENCE_TOP_K
        
        # 1. 转换 Tensor (使用外部传入的 Pos)
        src_tensor, pos_tensor, src_len = utils.token_to_tensor_v3_with_pos(
            melody_list, pos_list, self.melody_stoi, self.device
        )
        
        predicted_chords = []
        
        with torch.no_grad():
            # A. Encoder (传入 pos_tensor)
            encoder_outputs, hidden, cell = self.model.encoder(src_tensor, pos_tensor, src_len)
            
            # Bridge
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1).unsqueeze(0)
            cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1).unsqueeze(0)
            
            # Decoder Input
            sos_id = self.vocab['harmony'].get(config.SOS_TOKEN)
            trg_token = torch.tensor([sos_id], device=self.device)
            
            for step, input_token_str in enumerate(melody_list):
                output, hidden, cell = self.model.decoder(trg_token, hidden, cell, encoder_outputs)
                
                # BAR 逻辑
                if input_token_str == config.BAR_TOKEN:
                    bar_id = self.vocab['harmony'].get(config.BAR_TOKEN)
                    trg_token = torch.tensor([bar_id], device=self.device)
                    continue 

                logits = output / temperature
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=1)
                top_k_probs = torch.softmax(top_k_logits, dim=1)
                sample_idx = torch.multinomial(top_k_probs, num_samples=1).squeeze(1)
                top1 = top_k_indices.gather(1, sample_idx.unsqueeze(1)).squeeze(1)
                
                chord_str = self.harmony_itos.get(top1.item(), config.UNK_TOKEN)
                predicted_chords.append(chord_str)
                trg_token = top1
                
        return predicted_chords
    
if __name__ == "__main__":
    composer = AIComposer()
    test_melody = ['60', '_', '_', '_', config.BAR_TOKEN, '62', '_', '_', '_']
    print(f"\n🎵 Input: {test_melody}")
    result = composer.predict(test_melody)
    print(f"🎹 Output: {result}")