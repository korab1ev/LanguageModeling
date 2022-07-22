class BasicModel(nn.Module):
    def __init__(self, num_tokens=num_tokens, emb_size=16, hid_size=64):
        super(self.__class__, self).__init__()
        self.emb = nn.Embedding(num_tokens, emb_size)
        self.hid_size = hid_size
        self.num_tokens = num_tokens

        self.enc0 = nn.GRU(emb_size, hid_size, batch_first=True)

        self.dec_start = nn.Linear(hid_size, hid_size)
        self.dec0 = nn.GRUCell(emb_size, hid_size)
        self.hid_to_logits = nn.Linear(hid_size, num_tokens)

    def forward(self, x):
        """ predicts next state h_t """
        assert isinstance(x, Variable) and isinstance(x.data, torch.LongTensor)
        x_emb = self.emb(x)
        h_t, _ = self.enc0(x_emb) # [batch_size x [ MAX_LENGTH x hid_size ]] 

        next_logits = self.hid_to_logits(h_t)
        next_logp = F.log_softmax(next_logits, dim=-1)
        return next_logp
    
    def encode(self, seed_phrase : str):
        """
        :Takes seed phrase starting with a whitespace (e.g. ' Korab')
        :returns: initial state h0 of the decoder
        """
        x_sequence = [token_to_id[token] for token in seed_phrase]
        x_sequence = torch.tensor([x_sequence], dtype=torch.int64)  
        x_emb = self.emb(x_sequence)
        enc_seq, [last_state_but_not_really] = self.enc0(x_emb)
        # enc_seq: [batch, time, hid_size], last_state: [batch, hid_size]
        # enc_seq -> contains the output features (h_t) from the last layer of the GRU, for each t  
        # last_state -> last state h_t of encoder (h_0 for decoder)    
        last_state = enc_seq[torch.arange(len(enc_seq)), x_sequence.shape[1] - 1]
        dec_start = self.dec_start(last_state)
        return dec_start # returns initial decoder state h0

    def decode_inference(self, seed_phrase=' ', max_length=MAX_LENGTH, temperature=1.0):
        '''
        The function generates text given a phrase of length <= MAX_LENGTH.
        :param seed_phrase: prefix characters. The RNN is asked to continue the phrase
        :param max_length: maximum output length, including seed_phrase
        :param temperature: coefficient for sampling.  higher temperature produces more chaotic outputs,
                            smaller temperature converges to the single most likely output
        '''    
        x_sequence = [token_to_id[token] for token in seed_phrase]
        x_sequence = torch.tensor([x_sequence], dtype=torch.int64)
        # 1. feed the seed phrase, if any
        h0 = self.encode(seed_phrase) 
        hid_state = h0
        #start generating
        for _ in range(max_length - len(seed_phrase)):
            last_symbol = x_sequence[:, -1]
            last_emb = emb(last_symbol)
            # 2. find new state h_t
            new_dec_state = dec0(last_emb, hid_state)
            # 3. calculate the output logits
            logp_next = hid_to_logits(new_dec_state)
            p_next = F.softmax(logp_next / temperature, dim=-1).data.numpy()[0]            
            # 4. sample next token and push it back into x_sequence
            next_ix = np.random.choice(self.num_tokens, p=p_next)
            next_ix = torch.tensor([[next_ix]], dtype=torch.int64)
            x_sequence = torch.cat([x_sequence, next_ix], dim=1)

        return ''.join([tokens[ix] for ix in x_sequence.data.numpy()[0]])
