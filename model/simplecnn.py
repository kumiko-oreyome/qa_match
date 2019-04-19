import torch.nn as nn
import torch.nn.functional as F
import torch



class SimpleCNN(nn.Module):
    # kernel_sizes : tuple of kernel_size and numbers [(3,5)] means 5  of 3*emb_dim kernels
    def __init__(self,vocab,emb_dim,kernel_sizes): 
        super(SimpleCNN, self).__init__()

        self.emb_dim = emb_dim
        self.kernel_sizes = kernel_sizes

        self.emb = nn.Embedding(vocab.size(),emb_dim, padding_idx=vocab._encode_one(vocab.PAD))
        nn.init.uniform_(self.emb.weight, -0.001, 0.001)

        self.conv_actv = nn.ReLU()

        self.conv_layers =  nn.ModuleList()
        for kernel_size,kernel_number in kernel_sizes:
            conv = nn.Conv2d(1, kernel_number, ( kernel_size,emb_dim))
            self.conv_layers.append(conv)

    def get_hypers(self):
        return {'emb_dim':self.emb_dim,'kernel_sizes':self.kernel_sizes}


    def forward_question(self,question_batch):
        return self._forward_text(question_batch)
    def forward_answer(self,answer_batch):
        return self._forward_text(answer_batch)
    # text batch (N,max_len)
    def _forward_text(self,text_batch):
        assert len(text_batch.size()) == 2
        # text batch (N,max_len,emb_dim)
        embedding_out = self.emb(text_batch)
        # text batch (N,1,max_len,emb_dim)
        embedding_out = embedding_out.unsqueeze(1)
        kernel_maps = []
        for conv in self.conv_layers:
            # (N,kernel_num,H,1)
            kernel_map = conv(embedding_out)
            ks = kernel_map.size()
            # (N,kernel_num,H)
            kernel_map  = kernel_map.squeeze(3)
            kernel_map  = self.conv_actv(kernel_map)
            # N * kernel_num 
            out = F.max_pool1d(kernel_map, kernel_map.size(2)).squeeze(2)
            kernel_maps.append(out)
        # N * ?
        result = torch.cat(kernel_maps,1)
        return result
    

    
        



