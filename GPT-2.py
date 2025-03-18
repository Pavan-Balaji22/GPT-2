
import torch
import torch.nn as nn
from torch.nn import functional as F
import mlflow as mlflow
import math
from dataclasses import dataclass

# Hyper Parameters
lr = 1e-3
max_iter = 5000
step_iter = 500
eval_iter = 200

if torch.cuda.is_available():
    device = 'cuda'
# elif torch.backends.mps.is_available():
#     device = "mps"
else:
    device = "cpu"


torch.manual_seed(1337)

shakspheredata= open("../datasets/llm/shakeshpere.txt",mode="r",encoding="utf8").read()
vocab = sorted(list(set(shakspheredata)))

stoi = {k:v for v,k in enumerate(vocab)}
itos = {v:k for v,k in enumerate(vocab)}
encode = lambda x:[stoi[i] for i in x]
decode = lambda x: "".join([itos[i] for i in x])

#Data preparation
text = torch.tensor(encode(shakspheredata))
n = int(.9*len(text))
train = text[:n]
val = text[n:]



@torch.no_grad()
def esitimate_loss():
    out = {}
    model.eval()
    for mode in ["train","val"]:
        losses = 0
        for i in range(eval_iter):
            X,Y = get_batch(mode,GPTconfig())
            _,lossb = model(X,Y)
            losses = losses+lossb
        out[mode] = losses/eval_iter
    model.train()
    return out

def get_batch(split:str,config):
    data = train if split == "train" else val
    ix =  torch.randint(len(data) - config.block_size ,(config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y


class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embed % config.n_heads == 0
        self.config = config
        self.headsize = config.n_embed//config.n_heads
        self.c_attn = nn.Linear(config.n_embed,3 *config.n_embed ,bias = config.Bias)
        self.c_proj = nn.Linear(config.n_embed,config.n_embed ,bias = config.Bias)
        self.register_buffer("tril",torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size))
        self.attdropout = nn.Dropout(config.dropout)
        self.mhdropout = nn.Dropout(config.dropout)
    
    def forward(self,x):
        B,T,C = x.shape
        q,k,v = self.c_attn(x).split(self.config.n_embed,dim=-1)

        k = k.view(B,T,self.config.n_heads,self.headsize).transpose(1,2) # B,n_head,T,headsize
        q = q.view(B,T,self.config.n_heads,self.headsize).transpose(1,2)
        v = v.view(B,T,self.config.n_heads,self.headsize).transpose(1,2)

        wei = q @ k.transpose(-2,-1) * (1/math.sqrt(k.shape[-1])) # B,n_head,T,headsize X # B,n_head,headsize,T = B,n_head,T,T
        wei = wei.masked_fill(self.tril[:,:,:T,:T]==0,float('-inf'))
        wei = F.softmax(wei,dim=-1)
        wei = self.attdropout(wei)
        out = wei @ v #  B,n_head,T,T x # B,n_head,T,nheadsize = B,n_head,T,headsize
        out = out.transpose(1,2).contiguous().view(B,T,C)
        out = self.mhdropout(self.c_proj(out))

        return out
class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc =  nn.Linear(config.n_embed,4 *config.n_embed)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embed,config.n_embed)

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x
class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        # self.attention = Multihead() # using Causual attention insted for efficiency
        self.attn = CausalSelfAttention(config) # B,T,C
        self.mlp = MLP(config)
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.ln_2 = nn.LayerNorm(config.n_embed)
    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        out = x + self.mlp(self.ln_2(x))
        return out
    
class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        # print(nvocab,n_embed,block_size)
        self.transformer = nn.ModuleDict({
        "wte" : nn.Embedding(config.nvocab,config.n_embed),
        "wpe": nn.Embedding(config.block_size,config.n_embed),
        "h" :nn.ModuleList([Block(config) for x in range(config.n_layer)]),
        "ln_F":nn.LayerNorm(config.n_embed),
        
        })
        self.lm_head=nn.Linear(config.n_embed,config.nvocab)
         # B,T,vocab_size
    
    def forward(self,idx,target=None):
        
        B,T, = idx.shape
        tok_embed = self.transformer.wte(idx) # B,T,C
        pos_embed = self.transformer.wpe(torch.arange(T,device=device))
        x = tok_embed + pos_embed # B,T,C
        for Block in self.transformer.h:
            x = Block(x)
        x = self.transformer.ln_F(x) # B,T,C
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            B,T,C = logits.shape # B,T,vocab_size
            logits = logits.view(B*T,C)
            targets = target.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits,loss

    def generate(self,max_tokens:int,idx:torch.Tensor,config):
        for _ in range(max_tokens):
            idx_cond = idx[:, -config.block_size:]
            logits,loss = self(idx_cond) # B,T,C
            logits = logits[:,-1,:] #B,C Picking last time step
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,1)
            idx =torch.cat((idx,idx_next), dim=1)

        return idx

@dataclass
class GPTconfig:
    nvocab = len(vocab)#50257
    batch_size = 64
    block_size = 8
    max_iter = 5000
    step_iter = 500
    eval_iter = 200
    n_embed = 32
    n_heads = 4
    n_layer = 4
    dropout = 0.2
    Bias = False

model = GPT(GPTconfig())
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')


#Training
print(f"Start Training in {device}")
for i in range(max_iter):
    if i % step_iter== 0:
        eloss = esitimate_loss()
        print(f"Loss at {i}: Train loss: {eloss['train']} | Validation loss :{eloss['val']}")

    #Forward Pass
    xb,yb = get_batch('train',GPTconfig())
    
    logits,loss = model(xb,yb)
    
    #Backpass
    optimizer.zero_grad(set_to_none= True)
    loss.backward()
    optimizer.step()

print(F" Final loss: {loss.item():.4f}")

#Generation
print(decode(model.generate(1000,idx = torch.zeros((1,1),dtype=torch.long))[0].tolist()))
