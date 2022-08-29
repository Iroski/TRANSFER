from turtle import forward
from torch import nn
import torch 
from torch.functional import F
class EnhanceEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, _weight)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def embedding(self,input):
        return F.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

    def forward(self,input):
        method='identifier'
        final_embeddings=torch.FloatTensor().to(self.device)
        for tokens in input:
            final_embedding=torch.FloatTensor().to(self.device) #embedding for one sample, len 400
            for i in range(len(tokens)):
                token=tokens[i]
                if type(token)==int:
                    final_embedding=torch.cat((final_embedding,self.embedding(torch.LongTensor([token]).to(self.device))),dim=0)
                else:
                    if method=='default':
                        final_embedding=torch.cat((final_embedding,self.embedding(torch.LongTensor([token[2]]).to(self.device))),dim=0)
                    elif method=='identifier':
                        tmp_embedding=torch.FloatTensor().to(self.device)
                        for j in range(len(token[0])):
                            identifier=token[0][j]
                            tmp_embedding=torch.cat((tmp_embedding,self.embedding(torch.LongTensor([identifier]).to(self.device))),dim=0)
                        final_token=torch.mean(tmp_embedding,axis=0)*0.5+self.embedding(torch.LongTensor([token[2]]).to(self.device))*0.5
                        final_embedding=torch.cat((final_embedding,final_token),dim=0)
                    else:
                        pass #todo
            final_embeddings=torch.cat((final_embeddings,final_embedding.unsqueeze(0)),dim=0)
        return final_embeddings