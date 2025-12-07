import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout):
        super().__init__()
        # 1. 嵌入层：把数字 ID 变成向量
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        # 2. 双向 LSTM
        # bidirectional=True 会让 hidden_dim 翻倍
        self.rnn = nn.LSTM(emb_dim, hidden_dim, bidirectional=True, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_len):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))
        
        # Pack Sequence: 让 LSTM 忽略掉 Padding 的 0，加速并提高精度
        # (需要移到 CPU 处理长度，这是 PyTorch 的一个小特性)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.cpu(), batch_first=True, enforce_sorted=False)
        
        outputs, (hidden, cell) = self.rnn(packed_embedded)
        
        # Unpack: 变回 Tensor
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        # outputs: [batch, src_len, hidden_dim * 2] (包含了前向和后向的所有信息)
        # hidden: [2, batch, hidden_dim] (最后的隐藏状态)
        
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # 这是一个简单的线性层，用来计算"注意力分数"
        # 输入是 (Encoder输出 + Decoder当前状态)
        # Encoder输出是双向的(hidden*2)，Decoder是单向的(hidden*2)，所以输入是 hidden*4
        # *这里为了简化，我们将Decoder设为双向的大小，或者将Encoder输出投影
        # 这里的实现：我们将 Encoder 的双向 hidden 拼接，作为 context
        self.attn = nn.Linear((hidden_dim * 2) + (hidden_dim * 2), hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [1, batch, hidden_dim*2] (Decoder 上一刻的状态)
        # encoder_outputs: [batch, src_len, hidden_dim*2]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # 重复 hidden 以便和 encoder_outputs 对齐
        hidden = hidden.repeat(src_len, 1, 1).permute(1, 0, 2) 
        
        # 计算能量 (Energy)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        
        # 计算权重: [batch, src_len, 1]
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, dropout):
        super().__init__()
        self.output_dim = output_dim
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.attention = Attention(hidden_dim)
        
        # Decoder 的 LSTM 输入是：Embedding向量 + Attention Context向量
        self.rnn = nn.LSTM((hidden_dim * 2) + emb_dim, hidden_dim * 2, batch_first=True)
        
        self.fc_out = nn.Linear((hidden_dim * 2) + (hidden_dim * 2) + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_step, hidden, cell, encoder_outputs):
        # input_step: [batch, 1] (当前这一步输入的和弦 ID)
        input_step = input_step.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_step)) # [batch, 1, emb_dim]
        
        # 1. 计算 Attention
        # hidden[-1] 取最后一层的状态
        a = self.attention(hidden[-1].unsqueeze(0), encoder_outputs) # [batch, src_len]
        a = a.unsqueeze(1) # [batch, 1, src_len]
        
        # 2. 计算 Context (加权平均 Encoder 的输出)
        weighted = torch.bmm(a, encoder_outputs) # [batch, 1, hidden_dim*2]
        
        # 3. LSTM Step
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        # 4. 预测下一个词
        # 拼接 Embedding, Weighted Context, 和 LSTM Output
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=2))
        prediction = prediction.squeeze(1) # [batch, output_dim]
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, src_len, teacher_forcing_ratio=0.5):
        # src: [batch, src_len]
        # trg: [batch, trg_len]
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        # 存储所有时间步的输出
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # 1. Encoder 编码
        encoder_outputs, hidden, cell = self.encoder(src, src_len)
        
        # 处理双向 LSTM 的 hidden 状态以适配 Decoder
        # 简单做法：将双向的 layers 拼接或求和。这里我们为了对齐维度，直接把双向 hidden 拼接
        # Encoder Hidden: [2, batch, hidden] -> Decoder需 [1, batch, hidden*2]
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1).unsqueeze(0)
        cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1).unsqueeze(0)
        
        # 2. Decoder 解码
        # 第一个输入是 <SOS> token
        input_step = trg[:, 0]
        
        for t in range(1, trg_len):
            # 运行一步 Decoder
            output, hidden, cell = self.decoder(input_step, hidden, cell, encoder_outputs)
            
            # 存入结果
            outputs[:, t, :] = output
            
            # Teacher Forcing 机制
            # 有一定概率使用真实的下一个词作为输入，而不是模型预测的词
            # 这能加速训练收敛
            top1 = output.argmax(1) 
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            input_step = trg[:, t] if use_teacher_forcing else top1
            
        return outputs