class BasicModel(nn.Module):
    def __init__(self, num_tokens=num_tokens, emb_size=16, hid_size=64):
        super(self.__class__, self).__init__()
        self.emb = nn.Embedding(num_tokens, emb_size)
        self.rnn = nn.RNN(emb_size, hid_size, batch_first=True)
        self.hid_to_logits = nn.Linear(hid_size, num_tokens)

    def forward(self, x, h_prev=None):
        """ predicts next hidden state h_t """
        assert isinstance(x, Variable) and isinstance(x.data, torch.LongTensor)
        output, hn = self.rnn(self.emb(x), h_prev) # [batch_size x [ MAX_LENGTH x hid_size ]] 
        # output: containing the output features (h_t) from the last layer of the GRU, for each t
        # hn: tensor containing the final hidden state for the input sequence.
        next_logits = self.hid_to_logits(output)
        next_logp = F.log_softmax(next_logits, dim=-1)
        return next_logp, hn