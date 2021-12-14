import torch.nn as nn
import torch

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Forward function
        :param query: [batch size, sequence length, hidden dim]
        :param key: [batch size, sequence length, hidden dim]
        :param value: [batch size, sequence length, hidden dim]
        :param mask: Just pass None to mask. No need to handle it specifically.
        :return: [batch size, sequence length, hidden dim]
        """
        # Add your code here.
        batch_size, s_length, h_dim = query.size()
        # reshape to batch_size, sequence_length, num_heads, hidden_dim
        key = self.linear_layers[0](key).view(batch_size, s_length, self.h, self.d_k)
        query = self.linear_layers[1](query).view(batch_size, s_length, self.h, self.d_k)
        value = self.linear_layers[2](value).view(batch_size, s_length, self.h, self.d_k)
        # reshape to batch_size, num_heads, sequence_length, hidden_dim
        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)
        # get score
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_k**0.5
        # apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -9e15)
        # get attention
        attentions = nn.functional.softmax(scores, dim=-1)
        # apply dropout
        attentions = self.dropout(attentions)
        # get attention head outputs
        output = torch.matmul(attentions, value)
        # reshape to batch_size, sequence_length, num_heads, hidden_dim
        output = output.permute(0, 2, 1, 3)
        output = output.reshape(batch_size, s_length, h_dim)

        output = self.output_linear(output)

        return output


if __name__ == '__main__':
    # # test the module.
    # batch_size = 32
    # s_length = 12
    # h_dim = 10

    # data = torch.randn(batch_size, s_length, h_dim)

    # my_mha = MultiHeadedAttention(5, h_dim)
    # output = my_mha.forward(data, data, data)

    # print(output.size())
    pass
