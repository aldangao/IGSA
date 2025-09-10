#在文本攻击中实施梯度反向传播
import numpy as np 
import torch
import torch.nn as nn

import copy
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import random  # 导入random模块，用于生成随机数
from models import clip
from models.clip_model.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
class IGSAttacker():
    def __init__(self, model, img_attacker, txt_attacker,critic_model,args):
        self.model=model
        self.img_attacker = img_attacker
        self.txt_attacker = txt_attacker
        self.critic_model=critic_model
        self.steps = 10
        self.args = args
    
        def pgd_attack(self, imgs, texts, txt2img, device='cpu', max_length=30, scales=None, **kwargs):
        max_score_index = self.txt_attacker.get_important_scores_emb(imgs, texts, self.critic_model)
        # max_word = [texts[i][max_score_index[i]] for i in range(len(max_score_index))]
        masks_words,masks_tokens,max_word_mask = self.txt_attacker.get_max_word_mask(texts, max_score_index, emb_dim=self.model.text_encoder.embeddings.word_embeddings.embedding_dim, device=device)
        #数据准备
        ##text
        text_inputs = self.txt_attacker.tokenizer(texts, padding='max_length', truncation=True, max_length=self.txt_attacker.max_length, return_tensors='pt').to(device)
        word_emb = self.model.text_encoder.embeddings.word_embeddings(text_inputs.input_ids)  # batch, seq-len, dim
        # word_emb.requires_grad_()
        adv_text_emb = word_emb.detach() + torch.from_numpy(np.random.uniform(-self.txt_attacker.eps, self.txt_attacker.eps, word_emb.shape)).float().to(device) * max_word_mask
        adv_text_emb = torch.clamp(adv_text_emb, -3, 3)  
        ##img
        adv_imgs = imgs.detach() + torch.from_numpy(np.random.uniform(-self.img_attacker.eps, self.img_attacker.eps, imgs.shape)).float().to(device)
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
        b, _, _, _ = adv_imgs.shape
        #开始循环
        for i in range(self.steps):
            adv_text_emb.requires_grad_()
            adv_imgs.requires_grad_()
            adv_imgs_scale = self.get_scaled_imgs_1(adv_imgs, scales=scales, device=device)#注意区分1和2，2是5个对5个，1是5个对5个得25个
            self.model.zero_grad()
            self.critic_model.zero_grad()
            if adv_text_emb.grad is not None:
                adv_text_emb.grad.zero_()
            if adv_imgs.grad is not None:
                adv_imgs.grad.zero_()
            with torch.enable_grad():
                loss = -self.critic_model.fc_forward_emb(adv_imgs_scale, adv_text_emb, text_inputs.attention_mask, model=self.model)
                loss = loss.sum(dim=-1).mean()
            loss.backward()
            ##text embedding梯度反向传播
            grad = adv_text_emb.grad  # batch, seq-len, dim
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2), keepdim=True)
            perturbation = self.txt_attacker.stepsize * grad.sign()
            perturbation = perturbation * max_word_mask
            adv_text_emb = adv_text_emb.detach() + perturbation
            # adv_text_emb = torch.min(torch.max(adv_text_emb, word_emb -self.txt_attacker.eps), word_emb + self.txt_attacker.eps)
            adv_text_emb = torch.clamp(adv_text_emb, -3, 3)
            adv_text_emb = adv_text_emb.detach().requires_grad_(True)
            # adv_text_emb = adv_text_emb.detach()
            ##img梯度反向传播
            grad = adv_imgs.grad 
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)           
            perturbation = self.img_attacker.step_size * grad.sign()
            adv_imgs = adv_imgs.detach()+ perturbation
            adv_imgs = torch.min(torch.max(adv_imgs, imgs - self.img_attacker.eps), imgs + self.img_attacker.eps)
            adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
            adv_imgs = adv_imgs.detach().requires_grad_(True)

            torch.cuda.empty_cache()
        adv_text_id = self.txt_attacker.embedding_to_text_nearestneighbor(adv_text_emb, text_inputs, self.model)
        adv_text = [self.txt_attacker.tokenizer.decode(text_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for text_id in adv_text_id]

        return adv_imgs, adv_text
    def get_scaled_imgs_1(self, imgs, scales=None, device='cuda'):#abcd->abcdabcd
        if scales is None:
            return imgs

        ori_shape = (imgs.shape[-2], imgs.shape[-1])
        
        reverse_transform = transforms.Resize(ori_shape,
                                interpolation=transforms.InterpolationMode.BICUBIC)
        result = []
        for ratio in scales:
            scale_shape = (int(ratio*ori_shape[0]), 
                                  int(ratio*ori_shape[1]))
            scale_transform = transforms.Resize(scale_shape,
                                  interpolation=transforms.InterpolationMode.BICUBIC)
            scaled_imgs = imgs + torch.from_numpy(np.random.normal(0.0, 0.05, imgs.shape)).float().to(device)
            scaled_imgs = scale_transform(scaled_imgs)
            scaled_imgs = torch.clamp(scaled_imgs, 0.0, 1.0)
            
            reversed_imgs = reverse_transform(scaled_imgs)
            
            result.append(reversed_imgs)
        
        return torch.cat([imgs,]+result, 0)
    def get_scaled_imgs_2(self, imgs, scales=None, device='cuda'):#abcd->aabbccdd
        if scales is None:
            return imgs

        ori_shape = (imgs.shape[-2], imgs.shape[-1])
        
        reverse_transform = transforms.Resize(ori_shape,
                                interpolation=transforms.InterpolationMode.BICUBIC)
        
        result = []
        
        # 对每张图片单独处理
        for i in range(imgs.shape[0]):  # 遍历每张图片
            single_img = imgs[i:i+1]  # 取出单张图片，保持维度 [1, C, H, W]
            result.append(single_img)  # 先添加原图
            
            # 对当前图片应用所有scale变换
            for ratio in scales:
                scale_shape = (int(ratio*ori_shape[0]), 
                            int(ratio*ori_shape[1]))
                scale_transform = transforms.Resize(scale_shape,
                                interpolation=transforms.InterpolationMode.BICUBIC)
                
                scaled_img = single_img + torch.from_numpy(np.random.normal(0.0, 0.05, single_img.shape)).float().to(device)
                scaled_img = scale_transform(scaled_img)
                scaled_img = torch.clamp(scaled_img, 0.0, 1.0)
                
                reversed_img = reverse_transform(scaled_img)
                
                result.append(reversed_img)  # 添加处理后的图片
        
        return torch.cat(result, 0)
class ImageAttacker():
    def __init__(self, normalization, eps=2/255, steps=10, step_size=0.5/255):
        self.normalization = normalization
        self.eps = eps
        self.steps = steps 
        self.step_size = step_size 

###self.normalization
    def txt_guided_attack(self,imgs, texts,model,critic_model, device):
        # model = self.model
        # critic_model = self.critic_model,
        model.eval()
        critic_model.eval()

        adv_imgs = imgs.detach() + torch.from_numpy(np.random.uniform(-self.eps, self.eps, imgs.shape)).float().to(device)
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
        adv_imgs_copy = []
        for x in range(adv_imgs.shape[0]):
            imgs_x = adv_imgs[x].detach().unsqueeze(0)
            for i in range(self.steps):
                imgs_x.requires_grad_()
                model.zero_grad()
                adv_imgs_clone = adv_imgs.clone()
                adv_imgs_clone[x] = imgs_x
                model.zero_grad()
                critic_model.zero_grad()
                if imgs_x.grad is not None:
                    imgs_x.grad.zero_()
                with torch.enable_grad():
                    loss = -critic_model.PMI(adv_imgs_clone, texts,x)
                loss.backward()

                grad = imgs_x.grad 
                grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)           
                
                perturbation = self.step_size * grad.sign()
                imgs_x = imgs_x.detach()+ perturbation
                imgs_x = torch.min(torch.max(imgs_x, imgs[x] - self.eps), imgs[x] + self.eps)
                imgs_x = torch.clamp(imgs_x, 0.0, 1.0)
                torch.cuda.empty_cache()
            adv_imgs_copy.append(imgs_x)
        return adv_imgs
   

filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves', '.', '-', 'a the', '/', '?', 'some', '"', ',', 'b', '&', '!',
                '@', '%', '^', '*', '(', ')', "-", '-', '+', '=', '<', '>', '|', ':', ";", '～', '·']
filter_words = set(filter_words)
    

class TextAttacker():
    def __init__(self, ref_net, tokenizer, cls=True, max_length=30, number_perturbation=1, topk=10, threshold_pred_score=0.3, batch_size=32):
        self.ref_net = ref_net
        self.tokenizer = tokenizer
        self.max_length = max_length
        # epsilon_txt
        self.num_perturbation = number_perturbation
        self.threshold_pred_score = threshold_pred_score
        self.topk = topk
        self.batch_size = batch_size
        self.cls = cls
        self.eps = 0.3
        self.steps = 10
        self.stepsize = 0.03
        
        

    def pgd_attack(self, imgs, texts, net, critic_model, device):
        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(device)
        word_emb = net.text_encoder.embeddings.word_embeddings(text_inputs.input_ids)  # batch, seq-len, dim
        # all_emb = self.ref_net.embeddings(inputs_embeds=word_emb)  # batch, seq-len, dim
        # # text_emb = net.text_encoder(inputs_embeds=word_emb, attention_mask=text_inputs.attention_mask, mode='text').last_hidden_state  # batch, seq-len, dim
        # text_emb2 = net.text_encoder(text_inputs.input_ids, attention_mask=text_inputs.attention_mask, mode='text').last_hidden_state
        adv_text_emb = word_emb.detach() + torch.from_numpy(np.random.uniform(-self.eps, self.eps, word_emb.shape)).float().to(device)
        adv_text_emb = torch.clamp(adv_text_emb, -2, 2)  
        for i in range(self.steps):
            adv_text_emb.requires_grad_()
            critic_model.zero_grad()
            if adv_text_emb.grad is not None:
                adv_text_emb.grad.zero_()
            with torch.enable_grad():
                loss = -critic_model.fc_forward_emb(imgs, adv_text_emb, text_inputs.attention_mask)
            loss.backward()
            grad = adv_text_emb.grad  # batch, seq-len, dim
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2), keepdim=True)
            perturbation = self.stepsize * grad.sign()
            adv_text_emb = adv_text_emb.detach() + perturbation
            adv_text_emb = torch.min(torch.max(adv_text_emb, adv_text_emb -self.eps), adv_text_emb + self.eps)
            adv_text_emb = torch.clamp(adv_text_emb, -2, 2)
            adv_text_emb = adv_text_emb.detach()
        adv_text_id = self.embedding_to_text_nearestneighbor(adv_text_emb, text_inputs, net)
        adv_text = [self.tokenizer.decode(text_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for text_id in adv_text_id]

        return adv_text
    def embedding_to_text_nearestneighbor(self, adv_text_emb, text_inputs, net,args=None):
        """
        方案1: 通过最近邻搜索将embedding恢复为token ids
        """
        # 获取词汇表的embedding矩阵
        vocab_embeddings = net.weight #.text_encoder.embeddings.word_embeddings.weight  # [vocab_size, embedding_dim]
        # print('vocab_embeddings:')
        # print(vocab_embeddings.min().item(), vocab_embeddings.max().item(), vocab_embeddings.std().item())
        batch_size, seq_len, embed_dim = adv_text_emb.shape
        vocab_size = vocab_embeddings.shape[0]
        
        # 重塑为 [batch_size * seq_len, embed_dim] 以便计算
        adv_emb_flat = adv_text_emb.view(-1, embed_dim)  # [batch*seq, embed]
        
        # 计算与词汇表中每个词的cosine相似度
        # 方法1: 使用cosine相似度
        adv_emb_norm = F.normalize(adv_emb_flat, p=2, dim=1)  # [batch*seq, embed]
        vocab_emb_norm = F.normalize(vocab_embeddings, p=2, dim=1)  # [vocab, embed]
        
        # 计算相似度矩阵 [batch*seq, vocab]
        similarity = torch.mm(adv_emb_norm, vocab_emb_norm.t())
        
        # # 找到最相似的token
        # _, token_ids = torch.max(similarity, dim=1)  # [batch*seq]
        # token_ids = token_ids.view(batch_size, seq_len)  # [batch, seq]
        # 过滤掉 [unusedXXX] token
        if args.source_model in ['ALBEF', 'TCL']:#hasattr(self.tokenizer, "vocab"):
            # HuggingFace tokenizer
            valid_ids = [i for i, tok in enumerate(self.tokenizer.vocab.keys())
                        if not tok.startswith("[unused")]
        else:
            # CLIP tokenizer 没有 unused token
            valid_ids = list(range(vocab_size))

        # 只在 valid_ids 上取最大值
        similarity_filtered = similarity[:, valid_ids]
        max_idx = torch.argmax(similarity_filtered, dim=1)
        token_ids_flat = torch.tensor([valid_ids[i.item()] for i in max_idx], device=adv_text_emb.device)
        token_ids = token_ids_flat.view(batch_size, seq_len)

        if hasattr(text_inputs, "input_ids"):
            # 保持原始的特殊token (如 [CLS], [SEP], [PAD])
            original_ids = text_inputs.input_ids
            mask = text_inputs.attention_mask
            
            # 只替换非特殊token的位置
            special_tokens = {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, 
                            self.tokenizer.pad_token_id, self.tokenizer.mask_token_id}
            for i in range(batch_size):
                for j in range(seq_len):
                    if original_ids[i, j].item() in special_tokens or mask[i, j] == 0:
                        token_ids[i, j] = original_ids[i, j]
            adv_text = [self.tokenizer.decode(text_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for text_id in token_ids]
        else:
            original_ids = text_inputs
            mask = (text_inputs != 0).long()
            sot_token = _tokenizer.encoder["<|startoftext|>"]
            eot_token = _tokenizer.encoder["<|endoftext|>"]
            pad_token = 0
            special_tokens = {sot_token, eot_token, pad_token}
            for i in range(batch_size):
                for j in range(seq_len):
                    if original_ids[i, j].item() in special_tokens or mask[i, j] == 0:
                        token_ids[i, j] = original_ids[i, j]
            # adv_text = [_tokenizer.decode(text_id) for text_id in token_ids]
            adv_text = []
            for text_id in token_ids:
                tokens = [t.item() for t in text_id if t.item() not in (sot_token, eot_token, 0)]
                adv_text.append(_tokenizer.decode(tokens))
        # adv_text = [self.txt_attacker.tokenizer.decode(text_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for text_id in adv_text_id]
        return adv_text

    def _tokenize(self, text):
        words = text.split(' ')

        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = self.tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)

        return words, sub_words, keys

    def _get_masked(self, text):
        words = text.split(' ')
        len_text = len(words)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
        # list of words
        return masked_words
    
    def get_important_scores_emb(self, imgs , texts, critic_model, model):
        num = len(texts)
        imgs = imgs.repeat_interleave(5, dim=0)
        max_score_index = []
        for i in range(num):
            masked_words = self._get_masked(texts[i])
            masked_texts = [' '.join(words) for words in masked_words]  # list of text of masked words
            imgs_x = imgs[i]
            imgs_x_batch = imgs_x.unsqueeze(0).repeat(len(masked_texts), 1, 1, 1)
            with torch.no_grad():
                import_scores = -critic_model.fc_forward(imgs_x_batch, masked_texts, model=model)
            import_scores = import_scores.sum(dim=-1)
            max_score_index.append(import_scores.argmax().item())
            torch.cuda.empty_cache()
        return max_score_index

    def get_max_word_mask(self, texts, mask_index,args, emb_dim =768, device='cpu'):
        """
        获取每个文本中最大重要性分数的词的mask
        :param texts: list of texts
        :param mask_index: list of indices of the most important words
        :return: mask tensor
        """
        masks_words = []
        for i, text in enumerate(texts):
            words = text.split(' ')
            if mask_index[i] < len(words):
                word = words[mask_index[i]]
                masks_words.append(word)
            else:
                masks_words.append("")  # 处理索引越界的情况
        
        if args.source_model in ['ALBEF', 'TCL']:
        # 将词转换为token id（分别处理每个词）
            masks_tokens = []
            for word in masks_words:
                if word:  # 非空字符串
                    tokens = self.tokenizer(word, add_special_tokens=False, return_tensors='pt').to(device)['input_ids'][0]
                    masks_tokens.append(tokens)
                else:
                    masks_tokens.append(torch.tensor([]))  # 空tensor for empty words
            
            # 完整文本的tokenization
            texts_tok = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(device)
            input_ids = texts_tok.input_ids
        else:
            texts_tok = clip.tokenize(texts, context_length=self.max_length).to(device)
            input_ids = texts_tok  # 直接就是 token id
            masks_tokens = []
            for word in masks_words:
                if word:
                    tok = clip.tokenize([word], context_length=self.max_length).to(device)[0]
                    # 去掉 padding 和特殊符号，只保留中间有效token
                    valid = tok[(tok != 0) & (tok != 49406) & (tok != 49407)]
                    masks_tokens.append(valid)
                else:
                    masks_tokens.append(torch.tensor([]).to(device))
        
        batch_size, seq_len = input_ids.shape
        # emb_dim = self.model.text_encoder.embeddings.word_embeddings.embedding_dim  # 获取embedding维度
        
        # 创建掩码矩阵
        mask = torch.zeros(batch_size, seq_len, emb_dim).to(device)
        
        for batch_idx in range(batch_size):
            # 获取当前样本的目标tokens
            target_tokens = masks_tokens[batch_idx]
            
            # 如果目标tokens为空，跳过
            if len(target_tokens) == 0:
                continue
                
            # 在完整文本的tokens中查找目标tokens的位置
            full_tokens = input_ids[batch_idx]
            
            # 使用滑动窗口查找子序列
            target_len = len(target_tokens)
            found_positions = []
            
            for start_pos in range(seq_len - target_len + 1):
                # 检查是否匹配
                if torch.equal(full_tokens[start_pos:start_pos + target_len], target_tokens):
                    found_positions.extend(range(start_pos, start_pos + target_len))
                    break  # 找到第一个匹配就停止
            
            # 如果没有找到完全匹配，尝试逐个token匹配（处理子词情况）
            if not found_positions:
                for target_token in target_tokens:
                    positions = torch.where(full_tokens == target_token)[0]
                    found_positions.extend(positions.tolist())
            
            # 去重并排序
            found_positions = sorted(list(set(found_positions)))
            
            # 设置掩码
            for pos in found_positions:
                if pos < seq_len:
                    mask[batch_idx, pos, :] = 1.0
        
        return masks_words,masks_tokens,mask