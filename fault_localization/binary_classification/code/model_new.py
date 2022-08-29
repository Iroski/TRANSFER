from torch import nn
import torch
import torch.nn.functional as F


class BinaryClassifier(nn.Module):

	def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size):
		super(BinaryClassifier, self).__init__()
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.label_size = label_size
		self.activation = torch.tanh
		self.num_layers = 1

		self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=self.num_layers, bidirectional=True, batch_first=True)
		self.decoder = nn.Linear(hidden_dim * 2, self.label_size)

	def forward(self, inputs):
		lstm_out, hidden = self.encoder(inputs)
		lstm_out = torch.transpose(lstm_out, 1, 2)
		out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
		out = self.decoder(out)
		return out
