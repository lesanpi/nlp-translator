import torch


class AttnEncoder(torch.nn.Module):
    def __init__(self, input_size, embedding_size=100, hidden_size=100, n_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(input_size, embedding_size)
        self.gru = torch.nn.GRU(
            embedding_size, hidden_size, num_layers=n_layers, batch_first=True)

    def forward(self, input_sentences):
        embedded = self.embedding(input_sentences)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


MAX_LENGTH = 10


class AttnDecoder(torch.nn.Module):
    def __init__(self, input_size, embedding_size=100, hidden_size=100, n_layers=2, max_length=MAX_LENGTH):
        super().__init__()

        self.embedding = torch.nn.Embedding(input_size, embedding_size)
        self.gru = torch.nn.GRU(
            embedding_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.out = torch.nn.Linear(hidden_size, input_size)

        # attention
        self.attn = torch.nn.Linear(hidden_size + embedding_size, max_length)
        self.attn_combine = torch.nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, input_words, hidden, encoder_outputs):

        # sacamos los embeddings
        embedded = self.embedding(input_words)
        # calculamos los pesos de la capa de atención
        attn_weights = torch.nn.functional.softmax(
            self.attn(torch.cat((embedded.squeeze(1), hidden[0]), dim=1)))
        # re-escalamos los outputs del encoder con estos pesos
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        output = torch.cat((embedded.squeeze(1), attn_applied.squeeze(1)), 1)
        # aplicamos la capa de atención
        output = self.attn_combine(output)
        output = torch.nn.functional.relu(output)
        # a partir de aquí, como siempre. La diferencia es que la entrada a la RNN
        # no es directmanete el embedding sino una combinación del embedding
        # y las salidas del encoder re-escaladas
        output, hidden = self.gru(output.unsqueeze(1), hidden)
        output = self.out(output.squeeze(1))
        return output, hidden, attn_weights
