""" Onmt NMT Model base class definition """
import torch.nn as nn

class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    # def __init__(self, encoder, decoder):
    #     super(NMTModel, self).__init__()
    #     self.encoder = encoder
    #     self.decoder = decoder
    def __init__(self, encoder, decoder,gate_attn='yes'):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.gate_attn = gate_attn

    def forward(self, src, tgt, lengths, bptt=False, with_align=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        dec_in = tgt[:-1]  # exclude last target from inputs

        enc_state, h_s, lengths = self.encoder(src, lengths)
        # print(self.type)
        # if self.gate_attn=='no':
        #     if bptt is False:
        #         self.decoder.init_state(src, h_s, enc_state)
        #     dec_out, attns = self.decoder(dec_in, h_s, None,
        #                                   memory_lengths=lengths,
        #                                   with_align=with_align)
        #     return dec_out, attns
        # else:
        # print('gated_auxi')
        auxi_state, auxi_hs, lengths = self.encoder(src, lengths,type='auxi')
        if bptt is False:
            self.decoder.init_state(src, h_s, enc_state)
        dec_out, attns = self.decoder(dec_in, h_s, auxi_hs,
                                      memory_lengths=lengths,
                                      with_align=with_align,)
        return dec_out, attns

        # if bptt is False:
        #     self.decoder.init_state(src, h_s, enc_state)
        # dec_out, attns = self.decoder(dec_in, h_s,
        #                               memory_lengths=lengths,
        #                               with_align=with_align)
        # return dec_out, attns
    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
