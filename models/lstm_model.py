import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        # x: [batch, seq, hidden]
        # Calculate attention scores
        energy = self.v(torch.tanh(self.W(x))) # [batch, seq, 1]
        attention_weights = torch.softmax(energy, dim=1)
        
        # Weighted sum of hidden states
        context = torch.sum(x * attention_weights, dim=1) # [batch, hidden]
        return context

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, output_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.attention = TemporalAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch, seq, input_size]
        out, _ = self.lstm(x) # out: [batch, seq, hidden]
        
        # Apply Temporal Attention
        context = self.attention(out) # [batch, hidden]
        
        out = self.dropout(context)
        return self.fc(out)
