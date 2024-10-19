import os
import sys
sys.path.append('./HMPT/datasets')
import tqdm
import torch.nn as nn
import copy
import torch
import math
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from bert_processors.abstract_processor import BertProcessor, InputExample
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionwiseFeedForward(nn.Module):
     "Implements FFN equation."

     def __init__(self, d_model, d_ff, dropout=0.1):
         super(PositionwiseFeedForward, self).__init__()
         self.w_1 = nn.Linear(d_model, d_ff)
         self.w_2 = nn.Linear(d_ff, d_model)
         self.dropout = nn.Dropout(dropout)

     def forward(self, x):
         return self.w_2(self.dropout(F.relu(self.w_1(x))))

def attention(query, key, value, mask=None,mask1=None,mask2=None,Weight=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # query,key,value:torch.Size([30, 8, 10, 64])
    # decoder mask:torch.Size([30, 1, 9, 9])
    d_k = query.size(-1)
    key_ = key.transpose(-2, -1)  # torch.Size([30, 8, 64, 10])
    # torch.Size([30, 8, 10, 10])
    scores = torch.matmul(query, key_) / math.sqrt(d_k)
    # learn_att = 0
    if mask1 is not None and mask is not None:
        scores = torch.mul((1 + torch.mul(F.softmax(mask1.masked_fill(mask1 == 0, -1e9), dim=-2), mask1)),scores) * Weight
        scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), scores
    elif mask == None and mask1 is not None:
        scores = torch.mul((1 + torch.mul(F.softmax(mask1.masked_fill(mask1 == 0, -1e9), dim=-2), mask1)),scores) * Weight
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), scores
    elif mask is not None and mask1 == None:
        scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value),scores
    else:
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value),scores

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        #Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h  # 48=768//16
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(110)

        self.weight = nn.Parameter(torch.ones(111,111))
    def forward(self, query, key, value, mask=None, mask1=None,mask2=None):
        # query,key,value:torch.Size([2, 10, 768])
        # if mask is not None:
        #     # Same mask applied to all h heads.
        #     mask = mask.unsqueeze(1)
        nbatches = query.size(0)    #2
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]  # query,key,value:torch.Size([30, 8, 10, 64])
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, mask1=mask1,mask2=mask2,
                                          Weight = self.weight, dropout=self.dropout)
         # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(
                  nbatches, -1, self.h * self.d_k)
        ret = self.linears[-1](x)  # torch.Size([2, 10, 768])
        return ret,self.attn
#layer normalization [(cite)](https://arxiv.org/abs/1607.06450). do on

class customizedModule(nn.Module):
    def __init__(self):
        super(customizedModule, self).__init__()

    # linear transformation (w/ initialization) + activation + dropout
    def customizedLinear(self, in_dim, out_dim, activation=None, dropout=False):
        cl = nn.Sequential(nn.Linear(in_dim, out_dim))
        nn.init.xavier_uniform(cl[0].weight)
        nn.init.constant(cl[0].bias, 0)

        if activation is not None:
            cl.add_module(str(len(cl)), activation)
        if dropout:
            cl.add_module(str(len(cl)), nn.Dropout(p=self.args.dropout))

        return cl

class SublayerConnection(customizedModule):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.init_mBloSA()

    def init_mBloSA(self):
        self.g_W1 = self.customizedLinear(768, 768)
        self.g_W2 = self.customizedLinear(768, 768)
        self.g_b = nn.Parameter(torch.zeros(768))

        self.g_W1[0].bias.requires_grad = False
        self.g_W2[0].bias.requires_grad = False

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        ret = x + self.dropout(sublayer(self.norm(x))[0])
        return ret,sublayer(self.norm(x))[1]

class SublayerConnection1(customizedModule):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection1, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.init_mBloSA()

    def init_mBloSA(self):
        self.g_W1 = self.customizedLinear(768, 768)
        self.g_W2 = self.customizedLinear(768, 768)
        self.g_b = nn.Parameter(torch.zeros(768))

        self.g_W1[0].bias.requires_grad = False
        self.g_W2[0].bias.requires_grad = False

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        ret = x + self.dropout(sublayer(self.norm(x)))
        return ret

# Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network.
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn      #多头注意力机制
        self.feed_forward = feed_forward    #前向神经网络
        # self.sublayer = clones(SublayerConnection(size, dropout), 2)

        self.sublayer = SublayerConnection(size, dropout)
        self.sublayer1 = SublayerConnection1(size, dropout)

        self.size = size

    def forward(self, x, mask,mask1,mask2):
        "Follow Figure 1 (left) for connections."
        x, att= self.sublayer(x, lambda x: self.self_attn(x, x, x, mask,mask1,mask2))
        # torch.Size([30, 10, 512])
        ret = self.sublayer1(x, self.feed_forward)
        return ret,att

class s2tSA(customizedModule):
    def __init__(self, hidden_size):
        super(s2tSA, self).__init__()

        self.s2t_W1 = self.customizedLinear(hidden_size, hidden_size, activation=nn.ReLU())
        self.s2t_W = self.customizedLinear(hidden_size, hidden_size)

    def forward(self, x):
        """
        source2token self-attention module
        :param x: (batch, (block_num), seq_len, hidden_size)
        :return: s: (batch, (block_num), hidden_size)
        """

        # (batch, (block_num), seq_len, word_dim)
        f = self.s2t_W1(x)
        f = F.softmax(self.s2t_W(f), dim=-2)
        # (batch, (block_num), word_dim)
        s = torch.sum(f * x, dim=-2)
        return s
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask,mask1,mask2):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x,att = layer(x, mask,mask1,mask2)
        return self.norm(x),att

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=False, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim

        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):      #4*768*768
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class SectionOne(customizedModule):
    def __init__(self,device):
        super(SectionOne, self).__init__()
        self.Encoder1 = Encoder(EncoderLayer(768, MultiHeadedAttention(16, 768), PositionwiseFeedForward(768, 3072), 0.1), 1)
        self.Encoder2 = Encoder(EncoderLayer(768, MultiHeadedAttention(16, 768), PositionwiseFeedForward(768, 3072), 0.1), 1)
        self.device=device
        self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
        self.cls_token1 = nn.Parameter(torch.randn(1, 1, 768))

    def forward(self,section_feature, image1,missing_img_prompt_ss,missing_img_prompt_se):
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=4)
        cls_tokens1 = repeat(self.cls_token1, '1 n d -> b n d', b=4)
        global_text_1 = torch.cat((cls_tokens,missing_img_prompt_ss, missing_img_prompt_se, section_feature, image1), dim=1)
        global_text_2 = torch.cat((cls_tokens1,missing_img_prompt_ss, section_feature), dim=1)
        global_ouput_1, m11 = self.Encoder1(global_text_1, mask=None, mask1=None, mask2=None)
        global_ouput_2, m21 = self.Encoder2(global_text_2, mask=None, mask1=None, mask2=None)

        return global_ouput_1[:, 0, :],global_ouput_2[:, 0, :]

class SentenceFour(customizedModule):
    def __init__(self,device):
        super(SentenceFour, self).__init__()
        self.device = device
        self.s2tSA = s2tSA(768)
        self.section_mask_full_3 = torch.cat(((torch.triu(torch.ones(100, 100), diagonal=-1).to(self.device) - torch.triu(torch.ones(100, 100), diagonal=2).to(self.device)),torch.ones(100, 42).to(self.device)),dim=1)
        self.section_mask_full_5 = torch.cat(((torch.triu(torch.ones(100, 100), diagonal=-2).to(self.device) - torch.triu(torch.ones(100, 100), diagonal=3).to(self.device)),torch.ones(100, 42).to(self.device)),dim=1)
        self.section_mask_full_7 = torch.cat(((torch.triu(torch.ones(100, 100), diagonal=-3).to(self.device) - torch.triu(torch.ones(100, 100), diagonal=4).to(self.device)),torch.ones(100, 42).to(self.device)),dim=1)

        self.zeros = torch.ones(42, 142).to(self.device)
        self.Encoder3 = Encoder(EncoderLayer(768, MultiHeadedAttention(16, 768), PositionwiseFeedForward(768, 3072), 0.1), 1)
        self.Encoder4 = Encoder(EncoderLayer(768, MultiHeadedAttention(16, 768), PositionwiseFeedForward(768, 3072), 0.1), 1)
        self.Encoder5 = Encoder(EncoderLayer(768, MultiHeadedAttention(16, 768), PositionwiseFeedForward(768, 3072), 0.1), 1)
        self.Encoder6 = Encoder(EncoderLayer(768, MultiHeadedAttention(16, 768), PositionwiseFeedForward(768, 3072), 0.1), 1)
        self.Encoder7 = Encoder(EncoderLayer(768, MultiHeadedAttention(16, 768), PositionwiseFeedForward(768, 3072), 0.1), 1)
        self.init_mBloSA()
        self.cls_token1 = nn.Parameter(torch.randn(1, 1, 768))
        self.cls_token2 = nn.Parameter(torch.randn(1, 1, 768))
        self.cls_token3 = nn.Parameter(torch.randn(1, 1, 768))
        self.cls_token4 = nn.Parameter(torch.randn(1, 1, 768))
        self.cls_token5 = nn.Parameter(torch.randn(1, 1, 768))

    def init_mBloSA(self):
        self.f_W3 = self.customizedLinear(768 * 2, 768, activation=nn.ReLU())

    def forward(self, sentence_feature, image_one,global_image,missing_img_prompt_ee,missing_img_prompt_se):
        sentence_feature = self.f_W3(torch.cat([sentence_feature, global_image.unsqueeze(1).repeat(1, 100, 1)], dim=2))  # 4*8*768

        final_mask_3 = torch.cat((self.section_mask_full_3,self.zeros),dim=0).unsqueeze(0).unsqueeze(0).expand(4, 16, 142,142)
        final_mask_5 = torch.cat((self.section_mask_full_5,self.zeros),dim=0).unsqueeze(0).unsqueeze(0).expand(4, 16, 142,142)
        final_mask_7 = torch.cat((self.section_mask_full_7,self.zeros),dim=0).unsqueeze(0).unsqueeze(0).expand(4, 16, 142,142)

        cls_tokens1 = repeat(self.cls_token1, '1 n d -> b n d', b=4)
        cls_tokens2 = repeat(self.cls_token2, '1 n d -> b n d', b=4)
        cls_tokens3 = repeat(self.cls_token3, '1 n d -> b n d', b=4)
        cls_tokens4 = repeat(self.cls_token4, '1 n d -> b n d', b=4)
        cls_tokens5 = repeat(self.cls_token5, '1 n d -> b n d', b=4)

        global_text_1 = torch.cat((sentence_feature, image_one,missing_img_prompt_se,missing_img_prompt_ee,cls_tokens1), dim=1)
        global_text_2 = torch.cat((sentence_feature, image_one,missing_img_prompt_se,missing_img_prompt_ee,cls_tokens2), dim=1)
        global_text_3 = torch.cat((sentence_feature, image_one,missing_img_prompt_se,missing_img_prompt_ee,cls_tokens3), dim=1)
        global_text_4 = torch.cat((sentence_feature, image_one,missing_img_prompt_se,missing_img_prompt_ee,cls_tokens4), dim=1)
        global_text_5 = torch.cat((cls_tokens5,missing_img_prompt_ee, sentence_feature), dim=1)

        global_text_11,att11 = self.Encoder3(global_text_1, mask=None, mask1 = None, mask2=None)
        global_text_21,att21 = self.Encoder4(global_text_2, mask=final_mask_3,mask1 = None, mask2=None)
        global_text_31,att31 = self.Encoder5(global_text_3, mask=final_mask_5,mask1 = None, mask2=None)
        global_text_41,att41 = self.Encoder6(global_text_4, mask=final_mask_7,mask1 = None, mask2=None)
        global_text_51,att51 = self.Encoder7(global_text_5, mask=None,mask1 = None, mask2=None)

        u22 = torch.cat([global_text_11[:,-1,:].unsqueeze(1),global_text_21[:,-1,:].unsqueeze(1),global_text_31[:,-1,:].unsqueeze(1),global_text_41[:,-1,:].unsqueeze(1)], dim=1)

        return self.s2tSA(u22),global_text_51[:,0,:]

class AAPDProcessor(BertProcessor):
    NAME = 'AAPD'
    NUM_CLASSES = 7
    IS_MULTILABEL = False

    def get_train_examples(self, data_dir):
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir,'MMaterials', 'exMMaterials_train.tsv')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'MMaterials', 'exMMaterials_dev.tsv')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'MMaterials', 'exMMaterials_test.tsv')), 'test')

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):

            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
