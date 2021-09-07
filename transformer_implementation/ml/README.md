# Implementation of a Translator Using Transformer Encoder-Decoder Model
### Uses machine-learning; Backends: PyTorch, torchtext

When I looked to understand transformer-style models, I *implemented* a translator for English/German *from scratch using pyTorch*.

## Transformer:
The transformer follows the implementation presented in [Attention is all you need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf).
<br>
Instead of a ReLU, I use a GeLU as implemented in the BERT model; on average their performance seems to be better for corner cases.
