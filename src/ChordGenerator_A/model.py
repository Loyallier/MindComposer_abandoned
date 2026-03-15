import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout, pos_dim, pos_emb_dim):
        """
        [V3.1] Added pos_dim (positional vocabulary size) and pos_emb_dim (positional vector dimension)
        """
        super().__init__()
        # 1. Melody Embedding
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        # 2. [V3.1] Positional Embedding
        self.pos_embedding = nn.Embedding(pos_dim, pos_emb_dim)
        
        # 3. Bidirectional LSTM
        # Input dimension = Melody vector + Positional vector
        total_input_dim = emb_dim + pos_emb_dim
        self.rnn = nn.LSTM(total_input_dim, hidden_dim, bidirectional=True, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, pos, src_len):
        # src: [batch, seq_len]
        # pos: [batch, seq_len]
        
        # 1. Get both embeddings
        embedded_src = self.dropout(self.embedding(src))  # [batch, seq, emb_dim]
        embedded_pos = self.dropout(self.pos_embedding(pos)) # [batch, seq, pos_emb_dim]
        
        # 2. Concatenate -> [batch, seq, emb_dim + pos_emb_dim]
        combined_input = torch.cat((embedded_src, embedded_pos), dim=2)
        
        # 3. Pack Sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(
            combined_input, src_len.cpu(), batch_first=True, enforce_sorted=False
        )
        
        outputs, (hidden, cell) = self.rnn(packed_input)
        
        # Unpack
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Encoder bidirectional (hidden*2) + Decoder unidirectional (hidden*2)
        self.attn = nn.Linear((hidden_dim * 2) + (hidden_dim * 2), hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [1, batch, hidden_dim*2]
        # encoder_outputs: [batch, src_len, hidden_dim*2]
        src_len = encoder_outputs.shape[1]
        
        hidden = hidden.repeat(src_len, 1, 1).permute(1, 0, 2)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.attention = Attention(hidden_dim)
        self.rnn = nn.LSTM((hidden_dim * 2) + emb_dim, hidden_dim * 2, batch_first=True)
        self.fc_out = nn.Linear((hidden_dim * 2) + (hidden_dim * 2) + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_step, hidden, cell, encoder_outputs):
        input_step = input_step.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_step))
        
        a = self.attention(hidden[-1].unsqueeze(0), encoder_outputs)
        a = a.unsqueeze(1)
        
        weighted = torch.bmm(a, encoder_outputs)
        
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=2))
        prediction = prediction.squeeze(1)
        
        return prediction, hidden, cell, a.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, pos, trg, src_len, teacher_forcing_ratio=0.5):
        # [V3.1] Added pos parameter
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Pass pos to Encoder
        encoder_outputs, hidden, cell = self.encoder(src, pos, src_len)
        
        # Handle hidden/cell (Bidirectional -> Unidirectional concatenation)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1).unsqueeze(0)
        cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1).unsqueeze(0)
        
        input_step = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(input_step, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            
            top1 = output.argmax(1)
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            input_step = trg[:, t] if use_teacher_forcing else top1
            
        return outputs