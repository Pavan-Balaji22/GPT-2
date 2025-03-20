
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass
import tiktoken

# Device selection #

if torch.cuda.is_available():
    device = 'cuda'
# elif torch.backends.mps.is_available():
#     device = "mps"
else:
    device = "cpu"


torch.manual_seed(1337)

shakspheredata= open("shakeshpere.txt",mode="r",encoding="utf8").read()

tokenizer = tiktoken.get_encoding("gpt2")
encode = tokenizer.encode
decode = tokenizer.decode

n = int(.9*len(shakspheredata))
train = shakspheredata[:n]
val = shakspheredata[n:]

@torch.no_grad()
def esitimate_loss():
    out = {}
    model.eval()
    for mode in ["train","val"]:
        losses = 0
        for i in range(hconfig.eval_iter):
            X,Y = evalloader.get_batch(mode,hconfig)
            _,lossb = model(X,Y)
            losses = losses+lossb
        out[mode] = losses/hconfig.eval_iter
    model.train()
    return out

# Data loader #
class DataLoader:
    def __init__(self,B,T,data):
        self.tokens = torch.tensor(encode(data))
        self.B,self.T = B,T
        self.length = len(self.tokens)
        self.cur_pos = 0
        self.epoch = 0
        print(f"Total number of tokens:{self.length}")
        
    def get_batch(self):
        data = self.tokens[self.cur_pos:((self.B*self.T)+1+self.cur_pos)]
        x = data[:-1].view(self.B,self.T)
        y = data[1:].view(self.B,self.T)
        x, y = x.to(device), y.to(device)
        self.cur_pos += (self.B*self.T)+1
        if self.cur_pos > self.length:
            self.epoch += 1
            self.cur_pos = 0
        return x,y

# Required layers #
class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embed % config.n_heads == 0
        self.config = config
        self.headsize = config.n_embed//config.n_heads
        self.c_attn = nn.Linear(config.n_embed,3 *config.n_embed ,bias = config.Bias)
        self.c_proj = nn.Linear(config.n_embed,config.n_embed ,bias = config.Bias)
        self.c_proj.res_flag = 1
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
        self.c_proj.res_flag = 1

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
    
    # Define GPt @ architecture #    
class GPT(nn.Module):

    def __init__(self,config):
        self.config = config
        super().__init__()
        self.transformer = nn.ModuleDict({
        "wte" : nn.Embedding(config.nvocab,config.n_embed),
        "wpe": nn.Embedding(config.block_size,config.n_embed),
        "h" :nn.ModuleList([Block(config) for x in range(config.n_layer)]),
        "ln_F":nn.LayerNorm(config.n_embed),
        
        })
        self.lm_head=nn.Linear(config.n_embed,config.nvocab)
        
        #weight sharing between input and output transformation
        
        self.transformer.wte.weight = self.lm_head.weight
         # B,T,vocab_size

        self.apply(self._init_layers)
    def _init_layers(self,module):
        if isinstance(module,nn.Linear):
            std = 0.02
            if hasattr(module,"res_flag"):
                std = ( 2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight,0,std)
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module,nn.Embedding):
            nn.init.normal_(module.weight,0,0.2)
            
    def forward(self,idx,target=None):
        self.train()
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

    @torch.no_grad()
    def generate(self,max_tokens:int,idx:torch.Tensor,config):
        self.eval()
        for _ in range(max_tokens):
            idx_cond = idx[:, -config.block_size:]
            logits,loss = self(idx_cond) # B,T,C
            logits = logits[:,-1,:] #B,C Picking last time step as it is the predictoin of next letter
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,1)
            idx =torch.cat((idx,idx_next), dim=1)

        return idx

# Hyper Parameters
@dataclass
class GPTconfig:
    nvocab = tokenizer.n_vocab
    batch_size = 64
    block_size = 256
    lr = 2e-3
    max_iter = 50
    step_iter = 500
    eval_iter = 200
    n_embed = 32
    n_heads = 8
    n_layer = 4
    dropout = 0.2
    Bias = False
hconfig = GPTconfig()
model = GPT(hconfig)
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=hconfig.lr)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
trainloader = DataLoader(4,32,train)

#Training
print(f"Start Training in {device}")
for i in range(hconfig.max_iter):
    # if i % hconfig.step_iter== 0:
    #     eloss = esitimate_loss()
    #     print(f"Loss at {i}: Train loss: {eloss['train']} | Validation loss :{eloss['val']}")

    #Forward Pass
    xb,yb = trainloader.get_batch()
    logits,loss = model(xb,yb)
    print(f"loss at step {i} {loss.item()}")
    
    #Backpass
    optimizer.zero_grad(set_to_none= True)
    loss.backward()
    optimizer.step()

print(F" Final loss: {loss.item():.4f}")
import sys; sys.exit(0)

#Generation
print(decode(model.generate(1000,idx = torch.zeros((1,1),dtype=torch.long,device=device),config=hconfig)[0].tolist()))
