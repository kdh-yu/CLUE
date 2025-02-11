import torch
import torch.nn as nn
from CLIP import clip
from CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch.nn.functional as F

class CLUE(nn.Module):
    def __init__(self, 
                 token_length=32,
                 device='cuda'):
        super().__init__()
        self.tokenizer = _Tokenizer()
        self.model, _ = clip.load('ViT-L/14', device='cpu')
        for param in self.model.parameters():
            param.requires_grad = False
        self.embedding_dim = self.model.visual.proj.shape[1]
        self.token_length = token_length
        self.prompt_dependent = nn.Parameter(torch.rand(token_length, self.embedding_dim).to(device))
        self.prompt_independent = nn.Parameter(torch.rand(token_length, self.embedding_dim).to(device))
        self.device = device
        self.class_name = ['normal ', 'diseased ']
        self.eos_length = 1 + self.token_length + token_length + 2 + 1
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        ).cuda()
    
    def get_text_embedding(self, x: str):
        text_token = self.tokenizer.encode(x)
        text_embedding = self.model.token_embedding(torch.LongTensor(text_token).to(self.device))
        return text_embedding
    
    def get_sos(self):
        sos_token = self.tokenizer.encoder["<|startoftext|>"]
        sos_embedding = self.model.token_embedding(torch.LongTensor([sos_token]).to(self.device))
        return sos_embedding
        
    def get_eos(self):
        eos_token = self.tokenizer.encoder["<|endoftext|>"]
        eos_embedding = self.model.token_embedding(torch.LongTensor([eos_token]).to(self.device))
        return eos_embedding
    
    def get_image_condition(self, cls_token):
        img_feat = self.mlp(cls_token[:, -1, :].float()).unsqueeze(1)
        prompt = self.prompt_dependent.unsqueeze(0).cuda() + img_feat
        return prompt
    
    def forward(self, obj, image_features):
        image_feature = image_features[-1]
        prompt_final = []
        B = image_feature.shape[0]
        
        # Template
        SOS = self.get_sos().unsqueeze(0).expand(B, -1, -1).cuda()
        P_dependent = self.get_image_condition(image_feature).cuda()
        P_independent = self.prompt_independent.unsqueeze(0).expand(B, -1, -1).cuda()
        EOS = self.get_eos().unsqueeze(0).expand(B, -1, -1).cuda()
        
        # Normal
        CLS = self.get_text_embedding(self.class_name[0] + obj).unsqueeze(0).expand(B, -1, -1).cuda()
        prompt_tmp = torch.concat([SOS, CLS, P_independent, P_dependent, EOS], dim=1).cuda()
        prompt = torch.zeros(77, dtype=torch.long).cuda()
        prompt = self.model.token_embedding(prompt).unsqueeze(0).expand(B, -1, -1).cuda()
        for b in range(B):
            prompt[b, :prompt_tmp.shape[1], :] = prompt_tmp[b]
        prompt_feature = self.model.encode_prompt(prompt, prompt_tmp.shape[1])
        prompt_feature = prompt_feature / prompt_feature.norm(dim=1, keepdim=True)
        prompt_final.append(prompt_feature)
        
        # AbNormal
        CLS = self.get_text_embedding(self.class_name[1] + obj).unsqueeze(0).expand(B, -1, -1)
        prompt_tmp = torch.concat([SOS, CLS, P_independent, P_dependent, EOS], dim=1).cuda()
        prompt = torch.zeros(77, dtype=torch.long).cuda()
        prompt = self.model.token_embedding(prompt).unsqueeze(0).expand(B, -1, -1)
        for b in range(B):
            prompt[b, :prompt_tmp.shape[1], :] = prompt_tmp[b]
        prompt_feature = self.model.encode_prompt(prompt, self.eos_length)
        prompt_feature = prompt_feature / prompt_feature.norm(dim=1, keepdim=True)
        prompt_final.append(prompt_feature)
        
        prompt_final = torch.stack(prompt_final, dim=1)
        return prompt_final
    
if __name__ == '__main__':
    clue = CLUE()
    print(clue.model.visual.proj.shape)