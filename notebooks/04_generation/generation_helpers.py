#!/usr/bin/env python

import numpy as np
import partitura as pt
from torch import nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer


########################################## TOKENIZATION ##########################################

PITCH_DICT = {
    36:0, #Kick
    38:1, #Snare (Head)
    40:2, #Snare (Rim) 
    37:3, #Snare X-Stick
    48:4, #Tom 1
    50:5, #Tom 1 (Rim) 
    45:6, #Tom 2
    47:7, #Tom 2 (Rim) 
    43:8, #Tom 3 
    58:9, #Tom 3 (Rim)
    46:10, #HH Open (Bow) 
    26:11, #HH Open (Edge) 
    42:12, #HH Closed (Bow)
    22:13, #HH Closed (Edge)
    44:14, #HH Pedal
    49:15, #Crash 1
    55:16, #Crash 1
    57:17, #Crash 2
    52:18, #Crash 2 
    51:19, #Ride (Bow) 
    59:20, #Ride (Edge) 
    53:21, #Ride (Bell)
    'default':22
}


INV_PITCH_DICT_SIMPLE = {
    0:36, #Kick
    1:38, #Snare (Head)
    2:44, #Tom 1
    3:42, #HH Open (Bow) 
    4:49, #Crash 1
    5:51 #Ride (Bow) 
}

def pitch_DEcoder(pitch, inv_dict = INV_PITCH_DICT_SIMPLE):
    # 22 different instruments  in dataset, default 23rd class 
    a = 20
    try:
        a = inv_dict[pitch]
    except:
        pass
        # print("unknown instrument")
    return a

def time_DEcoder(time_enc, ppq):
    # base 2 encoding of time, starting at half note, ending at 128th
    time = 0
    for i, cl in enumerate(time_enc):
        if cl in [0, 1] and i < 5:
            time += cl * ppq * (2 **(1-i))  
    return time

def velocity_DEcoder(vel):
    
    return np.clip(vel * 16, 0, 127)

def tempo_DEcoder(tmp):
    # 8 classes of tempo between 60 and 180
    return np.clip(tmp*15 + 60, 60, 179) 

def DEtokenizer(token):
    
    onset_time = time_DEcoder(token[:7], ppq= 1 )
    pitch = pitch_DEcoder(token[7])
    velocity = velocity_DEcoder(token[8])
    return onset_time, pitch, velocity

def tokens_2_notearray(tokens):
    fields = [
            ("onset_sec", "f4"),
            ("duration_sec", "f4"),
            ("pitch", "i4"),
            ("velocity", "i4"),
        ]
    rows = []
    for token in tokens:
        onset_time, pitch, velocity = DEtokenizer(token)
        if pitch == 20:
            break
        else:
            rows.append((onset_time, 0.25, pitch, velocity))
    
    return np.array(rows, dtype=fields)
        
def save_notearray_2_midifile(na, no=0, fn = "test_beat"):
    pp = pt.performance.PerformedPart.from_note_array(na)
    pt.save_performance_midi(pp, fn+str(no)+".mid", mpq = 1000000)
    
    
    
########################################## DATASET GENERATION ##########################################


def generate_tokenized_data(seqs, 
                            measure_segmentation,
                            tokenizer,
                            minimal_notes = 1):
    SOS_token = np.array([[2,2,2,2,2,2,2, # time encoding
                         6,8,8,2]]) # instrument, velocity, tempo, beat/fill
    EOS_token = SOS_token + 1
    
    data = []
    segmented_seqs = list()
    
    for seq in seqs:
        segmented_seqs += measure_segmentation(seq, minimal_notes = minimal_notes)
    
    for s in segmented_seqs:
        tokens = np.array(tokenizer(s))
        tokens = np.concatenate((SOS_token, tokens, EOS_token), axis = 0)
        ## target_tokens = np.copy(tokens)
        data.append(tokens)

    # np.random.shuffle(data)

    return data


def batch_data(data, batch_size=16, padding=True, padding_token=-1):
    batches = []
    for idx in range(0, len(data), batch_size):
        if idx + batch_size < len(data):
            if padding:
                max_batch_length = 0

                # Get longest sentence in batch
                for seq in data[idx : idx + batch_size]:
                    if len(seq) > max_batch_length:
                        max_batch_length = len(seq)

                # Append X padding tokens until it reaches the max length
                for seq_idx in range(batch_size):
                    remaining_length = max_batch_length - len(data[idx + seq_idx])
                    data[idx + seq_idx] = np.concatenate((
                        data[idx + seq_idx],
                        np.ones((remaining_length,11))*np.array([[2,2,2,2,2,2,2, # time encoding
                         6,8,8,2]]) + 1
                        #np.full((remaining_length, 11), padding_token)
                    ))

            batches.append(np.array(data[idx : idx + batch_size]).astype(np.int64))

    print(f"{len(batches)} batches of size {batch_size}")
    return batches


########################################## MODEL ##########################################



class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-np.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])
    
    
class Transformer(nn.Module):
    """
    """
    # Constructor
    def __init__(
        self,
        tokens2dims,
        MultiEmbedding,
        num_heads,
        num_decoder_layers,
        dropout_p,
    ):
        super().__init__()
        
        self.tokens2dims = tokens2dims
        self.tokennumber_totaldim = np.array(self.tokens2dims).sum(0)
        self.dim_model = self.tokennumber_totaldim[1]

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=self.tokennumber_totaldim[1], dropout_p=dropout_p, max_len=5000
        )
        self.embedding = MultiEmbedding(tokens2dims) # multiembedding with dim 22
        
        # DECODER LAYERS
        D_layers = TransformerEncoderLayer(self.tokennumber_totaldim[1], 
                                         nhead = num_heads, 
                                         dim_feedforward = self.tokennumber_totaldim[1], 
                                         dropout=dropout_p)
        
        self.transformerDECODER = nn.TransformerEncoder(
            encoder_layer = D_layers,
            num_layers = num_decoder_layers,
        )
        self.out = nn.Linear(self.tokennumber_totaldim[1], self.tokennumber_totaldim[0])
        
    def forward(self, src, tgt_mask=None, tgt_pad_mask=None):

        src = self.embedding(src) * np.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        
        # to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1,0,2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformerDECODER(src=src, 
                                                  mask=tgt_mask, 
                                                  src_key_padding_mask = tgt_pad_mask)
        out = self.out(transformer_out)
        
        return out
      
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int = -1) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)
    

########################################## SAMPLING ##########################################

    
def sample_from_logits(pred, pred_dims):
    """
    """
    # print(pred.shape)
    out = list()
    for k in range(11):
        # batch / sequence / logits /
        # out.append(torch.argmax(pred[:,:,pred_dims[k]:pred_dims[k+1]], dim = -1, keepdim = True))
        out.append(
            torch.reshape(
                torch.multinomial(
                nn.functional.softmax(pred[:,-1,pred_dims[k]:pred_dims[k+1]], dim = -1),
                1),(-1, 1, 1)
            )
        ) 
    o = torch.cat(out, dim=-1)
    # print(o.shape)

    return o

def sample_loop(model, 
                tokens2dims,
                device):
    """
    """
    t2d = np.array(tokens2dims)
    pred_dims = np.concatenate(([0],np.cumsum(t2d[:,0])))
    model.eval()

    dataloader = [np.array([[[2,2,2,2,2,2,2, # time encoding
                             6,8,8,2]
                             #,
                             #[0,1,0,0,0,0,0, # time encoding
                             #0,8,3,1]
                            ]]) * np.ones((16,1,1))]
      
    for batch in dataloader:
        y = batch
        y = torch.LongTensor(y).to(device)

        # seq / batch / logits
        pred = model(y)
        # batch / sequence / logits
        pred = pred.permute(1, 0, 2) 
        sample = sample_from_logits(pred, pred_dims)
        prevY = y
        y = torch.cat((y,sample), dim=1)
        
        for i in range(30):
            pred = model(y)
            pred = pred.permute(1, 0, 2) 
            sample = sample_from_logits(pred, pred_dims)
            prevX = y
            # print(sample[0,:,:])
            y = torch.cat((y,sample[:,-1:,:]), dim=1)
        
    return y