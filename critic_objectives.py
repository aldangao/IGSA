import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import transforms


class MI_head(nn.Module):
    def __init__(self,model, device, images_normalize, tokenizer, embed_dim=768):
        super().__init__()
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.images_normalize = images_normalize
        self.embed_dim = embed_dim
        self.fc = nn.Sequential(nn.Linear(embed_dim, 1),
                                    nn.Softplus())

    def forward(self,images,texts):
        device=self.device
        images = images.to(device)
        images = self.images_normalize(images)
        texts_input = self.tokenizer(texts, padding='max_length', truncation=True, max_length=30,return_tensors="pt").to(device)
        
        with torch.no_grad():
            # 获取文本和图像的 embedding
            text_emb = self.model.inference_text(texts_input)['text_embed']#bs,30,768
            image_emb = self.model.inference_image(images)['image_embed']#bs,577,768
        text_att = texts_input.attention_mask.detach()
        encoder_att = torch.ones(image_emb.size()[:-1], dtype=torch.long).to(images.device)

        
        batch_size = images.size(0)
        # 重复图像 embedding
        image_emb_repeat = image_emb.unsqueeze(1).repeat(1, batch_size, 1, 1)  # (batch_size, batch_size, 577, 768)
        image_emb_repeat = image_emb_repeat.view(batch_size * batch_size, -1, self.embed_dim)  # (batch_size * batch_size, 577, 768)

        # 延展文本 embedding
        text_emb_expand = text_emb.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # (batch_size, batch_size, 30, 768)
        text_emb_expand = text_emb_expand.view(batch_size * batch_size, -1, self.embed_dim)  # (batch_size * batch_size, 30, 768)

        # 延展 attention mask
        text_att_expand = text_att.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, batch_size, 30)
        text_att_expand = text_att_expand.view(batch_size * batch_size, -1)  # (batch_size * batch_size, 30)
        
        encoder_att_repeat = encoder_att.unsqueeze(1).repeat(1, batch_size, 1)  # (batch_size, batch_size, 577)
        encoder_att_repeat = encoder_att_repeat.view(batch_size * batch_size, -1)  # (batch_size * batch_size, 577)

        with torch.no_grad():
            fusion_output = self.model.text_encoder(encoder_embeds=text_emb_expand.to(device),
                                            attention_mask=text_att_expand.to(device),
                                            encoder_hidden_states=image_emb_repeat.to(device),
                                            encoder_attention_mask=encoder_att_repeat.to(device),
                                            return_dict=True,
                                            mode='fusion')#(batch_size * batch_size, 30, 768)
        # 提取 CLS token 表示
        cls_features = fusion_output.last_hidden_state[:, 0, :]
        x = self.fc(cls_features) 
        return x
    
    def fc_forward(self,images,texts,model=None):
        device=self.device
        images = images.to(device)
        images = self.images_normalize(images)
        texts_input = self.tokenizer(texts, padding='max_length', truncation=True, max_length=30,return_tensors="pt").to(device)
        fusion_output = self.model.inference(images,texts_input)
        fusion_output = fusion_output["fusion_output"]
        cls_features = fusion_output[:, 0, :]
        x = self.fc(cls_features) 
        return x
    def fc_forward_emb(self,images,word_emb,text_inputs,model=None):#文本输入是embedding，图片仍是原始数据
        device=self.device
        attention_mask = text_inputs.attention_mask
        images = images.to(device)
        images = self.images_normalize(images)
        image_emb = self.model.inference_image(images)['image_embed']#bs,577,768

        text_emb = self.model.text_encoder(inputs_embeds=word_emb, attention_mask=attention_mask, mode='text').last_hidden_state

        encoder_att = torch.ones(image_emb.size()[:-1], dtype=torch.long).to(images.device)
        fusion_output = self.model.text_encoder(encoder_embeds=text_emb,
                                            attention_mask=attention_mask,
                                            encoder_hidden_states=image_emb,
                                            encoder_attention_mask=encoder_att,
                                            return_dict=True,
                                            mode='fusion').last_hidden_state[:, 0, :]#(batch_size * batch_size, 30, 768)
        x = self.fc(fusion_output)
        return x


    def learning_loss(self, images, texts):
        batch_size = images.size(0)
        scores = self.forward(images, texts)
        scores = scores.view(batch_size, batch_size)
        T0 = torch.diag(scores)
        T1 = scores

        T1_1 = T1.exp().sum(dim = 1)
        T1_2 = T1.exp().sum(dim = 0)
        lower_bound = []
        for x in range(batch_size):
            T1_3 = (T1_1[x]+T1_2[x]-T0[x])/((2*batch_size)-1)
            pmi = T0[x] - T1_3.log()
            lower_bound.append(pmi)
        lower_bound = torch.stack(lower_bound)
        return -lower_bound.mean()
    def learning_loss_club(self, images, texts):
        batch_size = images.size(0)
        scores = self.forward(images, texts)
        scores = scores.view(batch_size, batch_size)
        T0 = torch.diag(scores)
        T1 = scores

        T1_1 = T1.sum(dim = 1)
        T1_2 = T1.sum(dim = 0)
        lower_bound = []
        for x in range(batch_size):
            T1_3 = (T1_1[x]+T1_2[x]-T0[x])/((2*batch_size)-1)
            pmi = T0[x] - T1_3
            lower_bound.append(pmi)
        lower_bound = torch.stack(lower_bound)
        return -lower_bound.mean()
    def PMIs(self,images,texts):
        batch_size = images.size(0)
        scores = self.forward(images, texts)
        scores = scores.view(batch_size, batch_size)
        T0 = torch.diag(scores)
        T1 = scores

        T1_1 = T1.sum(dim = 1)
        T1_2 = T1.sum(dim = 0)
        lower_bound = []
        for x in range(batch_size):
            T1_3 = (T1_1[x]+T1_2[x]-T0[x])/((2*batch_size)-1)
            pmi = T0[x] - T1_3
            lower_bound.append(pmi)
        return lower_bound
    def PMI(self,images,texts,index):
        PMIs = self.PMIs(images,texts)
        return PMIs[index]
    
    def MI(self,images,texts):
        mis = self.PMIs(images,texts)
        mis = torch.stack(mis,dim=0)
        return mis.mean()
