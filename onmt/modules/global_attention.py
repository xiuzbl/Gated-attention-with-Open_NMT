'''Gated attention from the paper
    'Not All Attention Is Needed: Gated Attention Network for Sequence Data'
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.distributions import Bernoulli
from onmt.modules.sparse_activations import sparsemax
from onmt.utils.misc import aeq, sequence_mask

filename = open('intermedia output', 'w+')


class GlobalAttention(nn.Module):
    """Attention nn module that is responsible for computing the alignment vectors."""

    def __init__(self, dim, coverage=False, attn_type="dot",
                 attn_func="softmax"):
        super(GlobalAttention, self).__init__()
        self.dim = dim
        assert attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type (got {:s}).".format(
                attn_type))
        self.attn_type = attn_type
        assert attn_func in ["softmax", "sparsemax"], (
            "Please select a valid attention function.")
        self.attn_func = attn_func

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)

        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)
        self.linear_map = nn.Linear(dim * 2, 1)
        self.mlp_h = nn.Sequential(
            nn.Linear(dim,dim),
            nn.Tanh(),
            # nn.ReLU(inplace=True),
            nn.Linear(dim,1))
        self.sigmoid = nn.Sigmoid()
        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)
    def softmax(self,scores,g):
        v, _ = torch.max(scores, 0)
        # print(v)
        # if v[0]<=0:
        #     v,_ = torch.min(scores)
        scores = scores-v
        # print(v)
        # scores = torch.exp(scores * g)
        scores = torch.exp(scores) * g
        return scores

    def forward(self, h_t, h_s, auxi_hs,memory_lengths=None, coverage=None):
        """
        Args:
          auxi_hs = (FloatTensor): auxiliary output vectors ``(batch, src_len, dim)``
          h_t (FloatTensor): query vectors ``(batch, tgt_len, dim)``
          h_s (FloatTensor): source vectors ``(batch, src_len, dim)``
          memory_lengths (LongTensor): the source context lengths ``(batch,)``
          coverage (FloatTensor): None (not supported yet)

        Returns:
          (FloatTensor, FloatTensor):

          * Computed vector ``(tgt_len, batch, dim)``
          * Attention distribtutions for each query
            ``(tgt_len, batch, src_len)``
        """

        # one step input
        if h_t.dim() == 2:
            one_step = True
            h_t = h_t.unsqueeze(1)
        else:
            one_step = False

        batch, source_l, dim = h_s.size()
        batch_, target_l, dim_ = h_t.size()

        aeq(batch, batch_) #  Assert all the arguments have the same value.
        aeq(dim, dim_)
        aeq(self.dim, dim)

        if coverage is not None:
            batch_, source_l_ = coverage.size()
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        if coverage is not None:
            cover = coverage.view(-1).unsqueeze(1)
            h_s += self.linear_cover(cover).view_as(h_s)
            h_s = torch.tanh(h_s)

        print('h_t',h_t,file=filename)
        '''Main Modification'''
        # Expand the target hidden state to concatenate with the source hidden state.
        H_t = h_t.expand(-1,source_l,-1)
        # print(torch.isnan(H_t),file=filename)
        concat_h = torch.cat([auxi_hs,H_t],2).view(batch*source_l,dim*2) # (batch*source_l,dim*2)
        # print('whereis nan concat_h',torch.isnan(concat_h),file=filename)
        new_concat_h = self.linear_map(concat_h).view(batch,source_l,1)
        # print('whereis nan new_concat_h',torch.isnan(new_concat_h),file=filename)
        # Calculate the probability of the ouput of auxiliary network.
        p = self.sigmoid(new_concat_h) # (batch,source_l,1)

        # Get the distribution of gate which follows the Bernoulli distribution with probability p.
        # For trainning:
        print(p)
        G = RelaxedBernoulli(torch.tensor([9]).cuda(),p).sample() # hyperparameter--temperature. (batch,source_l,1)
        # print('G',G,file=filename)
        # print('G', G)
        # For testing:
        # G = Bernoulli(p)

        # e = MLP(h_s) to get the infomation of source hidden state.
        e = self.mlp_h(h_s)
        # print('e',e,file=filename)
        e = e.view(batch,source_l,1)

        # Calculate the alignment score.
        align_score = (self.softmax(e,G)).transpose(1,2) # align_score--(batch,1,source)
        # align_score = (G*torch.exp((e))).transpose(1,2) # align_score--(batch,1,source)
        print('score',align_score,file=filename)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=source_l)
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align_score.masked_fill_(~mask, -float('inf'))

        align_vectors = align_score / torch.sum(align_score,2,keepdim=True) # alpha--(batch,1,source_l)

        # Calculate the context vectors.
        c = torch.bmm(align_vectors,h_s) # context_vec--(batch,1,dim)

        # concatenate context vectors with the currenct target hidden state.
        concat_c = torch.cat([c, h_t], 2).view(batch * target_l, dim * 2)

        # Get the final output hidden state.
        attn_h = self.linear_out(concat_c).view(batch, target_l, dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = torch.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)
            # print('align_vectors', align_vectors.size())
            # Check output sizes
            batch_, dim_ = attn_h.size()
            aeq(batch, batch_)
            aeq(dim, dim_)
            batch_, source_l_ = align_vectors.size()
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()
            # Check output sizes
            target_l_, batch_, dim_ = attn_h.size()
            aeq(target_l, target_l_)
            aeq(batch, batch_)
            aeq(dim, dim_)
            target_l_, batch_, source_l_ = align_vectors.size()
            aeq(target_l, target_l_)
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        return attn_h, align_vectors
