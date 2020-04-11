## Gated attention with Open NMT

### Introduction

My main modification in the global_attention(/onmt/modules/global_attention.py) combined the idea of gated attention in the paper 'Not all attention is needed' with Luong global attention methods in the paper 'Effective Approaches to Attention-based Neural Machine Translation'. I also modified the interface function between the source code of Open NMT(encoder and decoder files) and the newly added content. I did not make many changes to the source code, which will allow me to compare other models later.

The current accuracy of this program is around 65%. I will continue to work on it to get a better performance. 

### 
