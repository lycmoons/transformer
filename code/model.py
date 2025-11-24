import torch
from torch import nn, Tensor
from modules import PositionEncoder, EncoderBlock, DecoderBlock
from utils import create_causal_mask


''' Transformer '''
class Transformer(nn.Module):
    def __init__(self, num_encoder_blocks: int, num_decoder_blocks: int,
                 src_vocab_size: int, tgt_vocab_size: int, model_size: int,
                 dropout: float, num_heads: int, ffn_size: int):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, model_size)
        self.src_position_encoder = PositionEncoder(dropout)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(model_size, num_heads, dropout, ffn_size) for _ in range(num_encoder_blocks)])

        self.tgt_embedding = nn.Embedding(tgt_vocab_size, model_size)
        self.tgt_position_encoder = PositionEncoder(dropout)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(model_size, num_heads, dropout, ffn_size) for _ in range(num_decoder_blocks)])

        self.linear_o = nn.Linear(model_size, tgt_vocab_size)

    # src_input: (batch_size, seq_len)
    # tgt_input: (batch_size, seq_len)
    # src_padding_mask, tgt_causal_mask: (batch_size, num_querys, num_pairs)
    def forward(self, src_input: Tensor, tgt_input: Tensor, src_padding_mask: Tensor = None, tgt_causal_mask: Tensor = None) -> Tensor:
        enc_output = self.src_position_encoder(self.src_embedding(src_input))
        for encoder_block in self.encoder_blocks:
            enc_output = encoder_block(enc_output, src_padding_mask)
        
        dec_output = self.tgt_position_encoder(self.tgt_embedding(tgt_input))
        for decoder_block in self.decoder_blocks:
            dec_output = decoder_block(dec_output, enc_output, tgt_causal_mask, src_padding_mask)

        return self.linear_o(dec_output)


    # src_input: (1, seq_len)
    # src_padding_mask: (1, 1, num_pairs)
    def predict(self, src_input: Tensor, src_padding_mask: Tensor,
                bos_id: int, eos_id: int, max_len: int):
        enc_output = self.src_position_encoder(self.src_embedding(src_input))
        for encoder_block in self.encoder_blocks:
            enc_output = encoder_block(enc_output, src_padding_mask.expand(1, max_len, max_len))

        ids = [bos_id]
        while len(ids) < max_len:
            seq_len = len(ids)
            tgt_input = torch.tensor(ids).reshape(1, seq_len).to(enc_output.device)
            tgt_causal_mask = create_causal_mask(seq_len).unsqueeze(0).to(enc_output.device)
            dec_output = self.tgt_position_encoder(self.tgt_embedding(tgt_input))
            for decoder_block in self.decoder_blocks:
                dec_output = decoder_block(dec_output, enc_output, tgt_causal_mask, src_padding_mask.expand(1, dec_output.shape[1], max_len))
            next_id = self.linear_o(dec_output)[0, -1, :].argmax().item()
            if next_id == eos_id:
                break
            ids.append(next_id)
        return ids[1:]