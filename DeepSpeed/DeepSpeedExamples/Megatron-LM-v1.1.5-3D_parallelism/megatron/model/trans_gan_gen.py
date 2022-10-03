from megatron import get_args
from megatron import mpu
from megatron.module import MegatronModule

import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import numpy as np
from .ViT_helper import DropPath, to_2tuple, trunc_normal_
from .diff_aug import DiffAugment
import torch.utils.checkpoint as checkpoint

import time

from .trans_gan_dis import Discriminator

from .utils import init_method_normal
from .utils import scaled_init_method_normal
from ..fp16 import FP16_Module

from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec


# hongyi (sep 12nd) it seems loss calculation is optional, let's pass None for now
# def GeneratorLoss(output, labels):
#     """ we try to come up with a fake output """
#     labels, loss_mask = labels[0], labels[1]

#     losses = mpu.vocab_parallel_cross_entropy(output.contiguous().float(), labels)
#     loss_mask = loss_mask.view(-1)
#     loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
#     return loss


class matmul(nn.Module):
    def __init__(self):
        super().__init__()
        #self.weights = None
        
    def forward(self, x1, x2):
       # s, b, dim1, dim2 = x1.shape[0], x1.shape[1], x1.shape[2], x1.shape[3]
       # x = F.linear(x1.reshape(s*b, dim1, dim2), x2.reshape(s*b, dim2, -1))
       # x = x.reshape(s,b,dim1, -1)
        x = x1@x2
        return x
        
def count_matmul(m, x, y):
    num_mul = x[0].numel() * x[1].size(-1)
    # m.total_ops += torch.DoubleTensor([int(num_mul)])
    m.total_ops += torch.DoubleTensor([int(0)])
    
class PixelNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=2, keepdim=True) + 1e-8)


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def DummyGenLoss(output, labels):
    """ From pretrain_gpt2:forward_step() """
    #labels, loss_mask = labels[0], labels[1] # we don't actually need label here

    args = get_args()
    #losses = mpu.vocab_parallel_cross_entropy(output.contiguous().float(), labels)
    #loss_mask = loss_mask.view(-1)
    #loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    #dis_net = Discriminator(args=args).to(torch.cuda.current_device())
    #dis_net = FP16_Module(dis_net)
    #fake_validity = dis_net(output)
    #loss = -torch.mean(fake_validity)
    #print(output.shape, labels.shape)
    if isinstance(output, tuple):
        loss = torch.mean(output[0])
    else:
        loss = torch.mean(output)#torch.zeros(1,).to(torch.cuda.current_device())
    #print(loss)
    return loss


def leakyrelu(x):
    return nn.functional.leaky_relu_(x, 0.2)

class CustomAct(nn.Module):
    def __init__(self, act_layer):
        super().__init__()
        if act_layer == "gelu":
            self.act_layer = gelu
        elif act_layer == "leakyrelu":
            self.act_layer = leakyrelu
        
    def forward(self, x):
        return self.act_layer(x)
        
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = CustomAct(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
#         self.noise_strength_1 = torch.nn.Parameter(torch.zeros([]))
#         self.noise_strength_2 = torch.nn.Parameter(torch.zeros([]))
    def forward(self, x):
#         x = x + torch.randn([x.size(0), x.size(1), 1], device=x.device) * self.noise_strength_1
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
#         x = x + torch.randn([x.size(0), x.size(1), 1], device=x.device) * self.noise_strength_2
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ParallelTransGANMlp(MegatronModule):
    def __init__(self, in_features, init_method, output_layer_init_method, 
                hidden_features=None, out_features=None, act_layer=gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features 
        #self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1 = mpu.ColumnParallelLinear(
                in_features,
                hidden_features,
                gather_output=False,
                init_method=init_method,
                skip_bias_add=True
            )
        self.act = CustomAct(act_layer)
        #self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2 = mpu.RowParallelLinear(
                hidden_features,
                in_features,
                input_is_parallel=True,
                init_method=output_layer_init_method,
                skip_bias_add=True
            )

        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
#         x = x + torch.randn([x.size(0), x.size(1), 1], device=x.device) * self.noise_strength_1
        time_s = time.time()
        x, _ = self.fc1(x)
        #torch.cuda.synchronize()
        time_e = time.time()
        #print(f"mlp 1 uses {time_e - time_s}", x.shape, self.in_features, self.hidden_features, self.out_features)
        x = self.act(x)
        x = self.drop(x)
#         x = x + torch.randn([x.size(0), x.size(1), 1], device=x.device) * self.noise_strength_2
        time_s = time.time()
        x, _ = self.fc2(x)
        #torch.cuda.synchronize()
        time_e = time.time()
        #print(f"mlp 2 uses {time_e - time_s}", x.shape, self.hidden_features, self.in_features)
        x = self.drop(x)
        return x


class ParallelTransGanAttention(MegatronModule):
    def __init__(self, dim, init_method, 
            output_layer_init_method,
            num_heads=4, 
            qkv_bias=False, 
            qk_scale=None, 
            attn_drop=0., 
            proj_drop=0., 
            window_size=16):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        #print(f"using head {num_heads}")
        #assert False
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5


        ### parallel trans-GAN setup ###
        world_size = mpu.get_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(dim,
                                                    world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            dim, num_heads)
        self.num_attention_heads_per_partition = mpu.divide(
            num_heads, world_size)

        self.qkv = mpu.ColumnParallelLinear(
                dim,
                3 * dim,
                gather_output=False,
                init_method=init_method)

        self.attn_drop = nn.Dropout(attn_drop)

        # self.proj connects to the self.dense in `ParallelSelfAttention`
        self.proj = mpu.RowParallelLinear(
            dim,
            dim,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True)

        self.proj_drop = nn.Dropout(proj_drop)
        self.mat = matmul()
        self.window_size = window_size
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        #print(f"coords shape: {coords.shape}")
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        #print(f"shape: {relative_coords.shape}")
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.noise_strength_1 = torch.nn.Parameter(torch.zeros([]))

        trunc_normal_(self.relative_position_bias_table, std=.02)
        
    def forward(self, x):
        # hidden_states: [sq, b, h]
        B, N, C = x.shape

        # hongyi: let's leave this alone for now
        x = (x + torch.randn([x.size(0), x.size(1), 1], device=x.device) * self.noise_strength_1).to(torch.float16) # (hongyi) new, not in the standard implementation
        
        qkv_raw, _ = self.qkv(x) # (hongyi) this seems to be contained in the implementation
  
        #print(f"using C: {C}")
        qkv = qkv_raw.reshape(
                B, N, 3, self.num_heads, int(C / mpu.get_model_parallel_world_size()) // self.num_heads # (hongyi) we adjust this line by MP size
                ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        attn = (self.mat(q, k.transpose(-2, -1))) * self.scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = self.mat(attn, v).transpose(
                            1, 2).reshape(
                                B, N, int(C/mpu.get_model_parallel_world_size())
                                ) # (hongyi) we adjust this line by MP size
        x, _ = self.proj(x) 
        x = self.proj_drop(x)
        return x
    
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, 
            qkv_bias=False, 
            qk_scale=None, 
            attn_drop=0., 
            proj_drop=0., 
            window_size=16):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mat = matmul()
        self.window_size = window_size
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.noise_strength_1 = torch.nn.Parameter(torch.zeros([]))

        trunc_normal_(self.relative_position_bias_table, std=.02)
        
    def forward(self, x):
        B, N, C = x.shape
        x = x + torch.randn([x.size(0), x.size(1), 1], device=x.device) * self.noise_strength_1 # (hongyi) new, not in the standard implementation
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # (hongyi) this seems to be contained in the implementation
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = self.mat(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    
    
class CustomNorm(nn.Module):
    def __init__(self, norm_layer, dim):
        super().__init__()
        self.norm_type = norm_layer
        if norm_layer == "ln":
            self.norm = nn.LayerNorm(dim)
        elif norm_layer == "bn":
            self.norm = nn.BatchNorm1d(dim)
        elif norm_layer == "in":
            self.norm = nn.InstanceNorm1d(dim)
        elif norm_layer == "pn":
            self.norm = PixelNorm(dim)
        
    def forward(self, x):
        if self.norm_type == "bn" or self.norm_type == "in":
            x = self.norm(x.permute(0,2,1)).permute(0,2,1)
            return x
        elif self.norm_type == "none":
            return x
        else:
            return self.norm(x)
        
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm, window_size=16):
        super().__init__()
        self.norm1 = CustomNorm(norm_layer, dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, window_size=window_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = CustomNorm(norm_layer, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


#class ParallelBlock(nn.Module):
class ParallelTransformerBlock(MegatronModule):
    def __init__(self, dim, num_heads, init_method, 
                output_layer_init_method,
                mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                drop=0., attn_drop=0., drop_path=0., 
                act_layer=gelu, norm_layer=nn.LayerNorm, window_size=16):
        super().__init__()
        self.norm1 = CustomNorm(norm_layer, dim)
        self.dim = dim
        self.attn = ParallelTransGanAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            attn_drop=attn_drop, 
            proj_drop=drop, 
            window_size=window_size,
            init_method=init_method, 
            output_layer_init_method=output_layer_init_method)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = CustomNorm(norm_layer, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ParallelTransGANMlp(
                            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                            init_method=init_method, 
                            output_layer_init_method=output_layer_init_method
                        )
    def forward(self, inputs):
        time_s = time.time()
        x, mask = inputs
        if isinstance(x, tuple):
            #print("&&&&&& inside tuple, x size: {}".format(x[0].size()))
            x = x[0]
        elif isinstance(x, torch.Tensor):
            #print("&&&&&&& tensor x size: {}".format(x.size()))
            pass
        else:
            raise NotImplementedError("Unrecognized x type ...")
        print(x.shape)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        #torch.cuda.synchronize()
        #print(f"layer use {time.time() - time_s} {self.dim}")
        return  (x, mask)


class PositionEmbedding(MegatronModule):
    def __init__(self, bottom_width, embedding_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, bottom_width, embedding_dim))

    def forward(self, inputs):
        x, mask = inputs
        if isinstance(x, tuple):
            x = x[0] + self.pos_embed
            #tempt_res = x[0] + self.pos_embed
            #list_x = list(x) # reconstruct x
            #list_x[0] = tempt_res
            #x = tuple(list_x)
        elif isinstance(x, torch.Tensor):
            x = x + self.pos_embed
        else:
            raise NotImplementedError("Unrecognized x data type ...")
        return (x, mask)

    
class StageBlock(nn.Module):
    def __init__(self, depth, dim, num_heads, 
        mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., 
        drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm, window_size=16):
        super().__init__()
        self.depth = depth
        models = [Block(
                        dim=dim, 
                        num_heads=num_heads, 
                        mlp_ratio=mlp_ratio, 
                        qkv_bias=qkv_bias, 
                        qk_scale=qk_scale,
                        drop=drop, 
                        attn_drop=attn_drop, 
                        drop_path=drop_path, 
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        window_size=window_size
                        ) for i in range(depth)]
        self.block = nn.Sequential(*models)
    def forward(self, x):
        x = self.block(x)
        return x

# hongyi: we actually do not need this, let's flatten this in the training section
# class ParallelTransGANStageBlock(nn.Module):
#     def __init__(self, depth, dim, num_heads, 
#         mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., 
#         drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm, window_size=16):
#         super().__init__()
#         self.depth = depth
#         models = [StageBlock(
#                         dim=dim, 
#                         num_heads=num_heads, 
#                         mlp_ratio=mlp_ratio, 
#                         qkv_bias=qkv_bias, 
#                         qk_scale=qk_scale,
#                         drop=drop, 
#                         attn_drop=attn_drop, 
#                         drop_path=drop_path, 
#                         act_layer=act_layer,
#                         norm_layer=norm_layer,
#                         window_size=window_size
#                         ) for i in range(depth)]
#         self.block = nn.Sequential(*models)
#     def forward(self, x):
#         x = self.block(x)
#         return x


def pixel_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0,2,1)
    return x, H, W

def bicubic_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    #x = nn.functional.interpolate(x, scale_factor=2, mode='bicubic')
    x = nn.functional.interpolate(x, scale_factor=2)
    B, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0,2,1)
    return x, H, W

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Generator(nn.Module):
    def __init__(self, args, img_size=224, patch_size=16, in_chans=3, num_classes=10, embed_dim=384, depth=5,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super(Generator, self).__init__()
        self.args = args
        self.ch = embed_dim
        self.bottom_width = args.bottom_width
        self.embed_dim = embed_dim = args.gf_dim
        self.window_size = args.g_window_size
        norm_layer = args.g_norm
        mlp_ratio = args.g_mlp
        depth = [int(i) for i in args.g_depth.split(",")]
        act_layer = args.g_act
        self.l2_size = 0
        
        if self.l2_size == 0:
            self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.embed_dim)
        elif self.l2_size > 1000:
            self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.l2_size//16)
            self.l2 = nn.Sequential(
                        nn.Linear(self.l2_size//16, self.l2_size),
                        nn.Linear(self.l2_size, self.embed_dim)
                      )
        else:
            self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.l2_size)
            self.l2 = nn.Linear(self.l2_size, self.embed_dim)
        self.pos_embed_1 = nn.Parameter(torch.zeros(1, self.bottom_width**2, embed_dim))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, (self.bottom_width*2)**2, embed_dim))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, (self.bottom_width*4)**2, embed_dim))
        self.pos_embed_4 = nn.Parameter(torch.zeros(1, (self.bottom_width*8)**2, embed_dim//4))
        self.pos_embed_5 = nn.Parameter(torch.zeros(1, (self.bottom_width*16)**2, embed_dim//16))
        self.pos_embed_6 = nn.Parameter(torch.zeros(1, (self.bottom_width*32)**2, embed_dim//64))
                                        
        self.pos_embed = [
            self.pos_embed_1,
            self.pos_embed_2,
            self.pos_embed_3,
            self.pos_embed_4,
            self.pos_embed_5,
            self.pos_embed_6
        ]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth[0])]  # stochastic depth decay rule
        self.blocks_1 = StageBlock(
                            depth=depth[0],
                            dim=embed_dim, 
                            num_heads=num_heads, 
                            mlp_ratio=mlp_ratio, 
                            qkv_bias=qkv_bias, 
                            qk_scale=qk_scale,
                            drop=drop_rate, 
                            attn_drop=attn_drop_rate, 
                            drop_path=0,
                            act_layer=act_layer,
                            norm_layer=norm_layer,
                            window_size=8
                        )
        self.blocks_2 = StageBlock(
                            depth=depth[1],
                            dim=embed_dim, 
                            num_heads=num_heads, 
                            mlp_ratio=mlp_ratio, 
                            qkv_bias=qkv_bias, 
                            qk_scale=qk_scale,
                            drop=drop_rate, 
                            attn_drop=attn_drop_rate, 
                            drop_path=0, 
                            act_layer=act_layer,
                            norm_layer=norm_layer,
                            window_size=16
                        )
        self.blocks_3 = StageBlock(
                            depth=depth[2],
                            dim=embed_dim, 
                            num_heads=num_heads, 
                            mlp_ratio=mlp_ratio, 
                            qkv_bias=qkv_bias, 
                            qk_scale=qk_scale,
                            drop=drop_rate, 
                            attn_drop=attn_drop_rate, 
                            drop_path=0, 
                            act_layer=act_layer,
                            norm_layer=norm_layer,
                            window_size=32
                        )
        self.blocks_4 = StageBlock(
                            depth=depth[3],
                            dim=embed_dim//4, 
                            num_heads=num_heads, 
                            mlp_ratio=mlp_ratio, 
                            qkv_bias=qkv_bias, 
                            qk_scale=qk_scale,
                            drop=drop_rate, 
                            attn_drop=attn_drop_rate, 
                            drop_path=0, 
                            act_layer=act_layer,
                            norm_layer=norm_layer,
                            window_size=self.window_size
                        )
        self.blocks_5 = StageBlock(
                            depth=depth[4],
                            dim=embed_dim//16, 
                            num_heads=num_heads, 
                            mlp_ratio=mlp_ratio, 
                            qkv_bias=qkv_bias, 
                            qk_scale=qk_scale,
                            drop=drop_rate, 
                            attn_drop=attn_drop_rate, 
                            drop_path=0, 
                            act_layer=act_layer,
                            norm_layer=norm_layer,
                            window_size=self.window_size
                        )
        self.blocks_6 = StageBlock(
                            depth=depth[5],
                            dim=embed_dim//64, 
                            num_heads=num_heads, 
                            mlp_ratio=mlp_ratio, 
                            qkv_bias=qkv_bias, 
                            qk_scale=qk_scale,
                            drop=drop_rate, 
                            attn_drop=attn_drop_rate, 
                            drop_path=0, 
                            act_layer=act_layer,
                            norm_layer=norm_layer,
                            window_size=self.window_size
                        )
                                        
        for i in range(len(self.pos_embed)):
            trunc_normal_(self.pos_embed[i], std=.02)
        self.deconv = nn.Sequential(
            nn.Conv2d(self.embed_dim//64, 3, 1, 1, 0)
        )

    def forward(self, z, epoch):
        if self.args.latent_norm:
            latent_size = z.size(-1)
            z = (z/z.norm(dim=-1, keepdim=True) * (latent_size ** 0.5))
        if self.args.latent_norm:
            latent_size = z.size(-1)
            z = (z/z.norm(dim=-1, keepdim=True) * (latent_size ** 0.5))
        if self.l2_size == 0:
            x = self.l1(z).view(-1, self.bottom_width ** 2, self.embed_dim)
        elif self.l2_size > 1000:
            x = self.l1(z).view(-1, self.bottom_width ** 2, self.l2_size//16)
            x = self.l2(x)
        else:
            x = self.l1(z).view(-1, self.bottom_width ** 2, self.l2_size)
            x = self.l2(x)
            
        x = x + self.pos_embed[0]
        B = x.size()
        H, W = self.bottom_width, self.bottom_width
        x = self.blocks_1(x)
        
        # print("!!!!!!!!!!! H1: {}, W1: {}".format(H, W)) # H1: 8, W1: 8
        x, H, W = bicubic_upsample(x, H, W)
        # print("!!!!!!!!!!! H2: {}, W2: {}".format(H, W)) # H2: 16, W2: 16
        x = x + self.pos_embed[1]
        B, _, C = x.size()
        # print("B1 : {}, C1: {}".format(B, C)) # B1: 1, C1: 1024
        x = self.blocks_2(x)
        
        # print("!!!!!!!!!!! H2: {}, W2: {}".format(H, W)) # H2: 16, W2: 16
        x, H, W = bicubic_upsample(x, H, W)
        # print("!!!!!!!!!!! H3: {}, W3: {}".format(H, W)) # H3: 32, W3: 32
        x = x + self.pos_embed[2]
        B, _, C = x.size()
        # print("B2 : {}, C2: {}".format(B, C)) # B2 : 1, C2: 1024
        x = self.blocks_3(x)
        

        # print("!!!!!!!!!!! H3: {}, W3: {}".format(H, W)) # H3: 32, W3: 32
        x, H, W = pixel_upsample(x, H, W)
        # print("!!!!!!!!!!! H4: {}, W4: {}".format(H, W)) # H4: 64, W4: 64
        x = x + self.pos_embed[3]
        B, _, C = x.size()
        # print("B3 : {}, C3: {}".format(B, C)) # B3 : 1, C3: 256
        x = x.view(B, H, W, C)
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size*self.window_size, C)
        x = self.blocks_4(x)
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W).view(B,H*W,C)
            
        # print("!!!!!!!!!!! H4: {}, W4: {}".format(H, W)) # H4: 64, W4: 64
        x, H, W = pixel_upsample(x, H, W)
        # print("!!!!!!!!!!! H5: {}, W5: {}".format(H, W)) # H5: 128, W5: 128
        x = x + self.pos_embed[4]
        B, _, C = x.size()
        # print("B4 : {}, C4: {}".format(B, C)) # B4 : 1, C4: 64
        x = x.view(B, H, W, C)
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size*self.window_size, C)
        x = self.blocks_5(x)
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W).view(B,H*W,C)
                                        
        # print("!!!!!!!!!!! H5: {}, W5: {}".format(H, W)) # H5: 128, W5: 128
        x, H, W = pixel_upsample(x, H, W)
        # print("!!!!!!!!!!! H6: {}, W6: {}".format(H, W)) # H6: 256, W6: 256
        x = x + self.pos_embed[5]
        B, _, C = x.size()
        x = x.view(B, H, W, C)
        # print("B5: {}, C5: {}".format(B, C)) @@@ B5: 1, C5: 16
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size*self.window_size, C)
        x = self.blocks_6(x)
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W).view(B,H,W,C).permute(0,3,1,2)
        
        
        output = self.deconv(x)
        return output


class ParallelTransGANGeneratorPipe(PipelineModule,MegatronModule):
    def __init__(self, args, img_size=224, patch_size=16, in_chans=3, num_classes=10, embed_dim=384, depth=5,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, topology=None
                 ):
        #super(Generator, self).__init__()
        self.args = args
        self.ch = embed_dim
        self.bottom_width = args.bottom_width
        self.embed_dim = embed_dim = args.gf_dim
        self.window_size = args.g_window_size
        norm_layer = args.g_norm
        mlp_ratio = args.g_mlp
        depth = [int(i) for i in args.g_depth.split(",")]
        act_layer = args.g_act
        self.l2_size = 0
        
        self.bs = args.batch_size
        self.init_method = init_method_normal(args.init_method_std)
        self.output_layer_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)

        self.specs = []
        # if self.l2_size == 0:
        #     self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.embed_dim)
        # elif self.l2_size > 1000:
        #     self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.l2_size//16)
        #     self.l2 = nn.Sequential(
        #                 nn.Linear(self.l2_size//16, self.l2_size),
        #                 nn.Linear(self.l2_size, self.embed_dim)
        #               )
        # else:
        #     self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.l2_size)
        #     self.l2 = nn.Linear(self.l2_size, self.embed_dim)

        # append l1 hongyi: we hard code this because the original trans-gan implementation set it as 0
        #self.specs.append(lambda x: print("PPPPPPPPPPPPPPPPPPPP lambda ***** x size: {}".format(x.size())))
        self.specs.append(LayerSpec(nn.Linear,
                    args.latent_dim,
                    (self.bottom_width ** 2) * self.embed_dim
                ))
        #self.specs.append(lambda x: print("PPPPPPPPPPPPPPPPPPPP lambda ***** x size: {}".format(x.size())))
        self.specs.append(lambda x: (x.view(-1, self.bottom_width ** 2, self.embed_dim),
                                     torch.zeros((100, 100),
                                    dtype=torch.bool,
                                    device=torch.cuda.current_device()))) # transfer this part of code .view(-1, self.bottom_width ** 2, self.embed_dim)

        # self.pos_embed_1 = nn.Parameter(torch.zeros(1, self.bottom_width**2, embed_dim))
        # self.pos_embed_2 = nn.Parameter(torch.zeros(1, (self.bottom_width*2)**2, embed_dim))
        # self.pos_embed_3 = nn.Parameter(torch.zeros(1, (self.bottom_width*4)**2, embed_dim))
        # self.pos_embed_4 = nn.Parameter(torch.zeros(1, (self.bottom_width*8)**2, embed_dim//4))
        # self.pos_embed_5 = nn.Parameter(torch.zeros(1, (self.bottom_width*16)**2, embed_dim//16))
        # self.pos_embed_6 = nn.Parameter(torch.zeros(1, (self.bottom_width*32)**2, embed_dim//64))
                                        
        # self.pos_embed = [
        #     self.pos_embed_1,
        #     self.pos_embed_2,
        #     self.pos_embed_3,
        #     self.pos_embed_4,
        #     self.pos_embed_5,
        #     self.pos_embed_6
        # ]

        self.specs.append(
            LayerSpec(
                PositionEmbedding,
                bottom_width=self.bottom_width**2, 
                embedding_dim=embed_dim
            )
        ) #self.specs.append(lambda x: x + self.pos_embed[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth[0])]  # stochastic depth decay rule
        # self.blocks_1 = StageBlock(
        #                     depth=depth[0],
        #                     dim=embed_dim, 
        #                     num_heads=num_heads, 
        #                     mlp_ratio=mlp_ratio, 
        #                     qkv_bias=qkv_bias, 
        #                     qk_scale=qk_scale,
        #                     drop=drop_rate, 
        #                     attn_drop=attn_drop_rate, 
        #                     drop_path=0,
        #                     act_layer=act_layer,
        #                     norm_layer=norm_layer,
        #                     window_size=8
        #                 )

        for _ in range(depth[0]):
            self.specs.append(
                    LayerSpec(ParallelTransformerBlock,
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, 
                        qk_scale=qk_scale,
                        drop=drop_rate, 
                        attn_drop=attn_drop_rate, 
                        drop_path=0, 
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        window_size=self.bottom_width,
                        init_method=self.init_method,
                        output_layer_init_method=self.output_layer_init_method
                    )
                )
        self.specs.append(lambda x: (nn.functional.interpolate(x[0], scale_factor=1), x[1]))
        self.specs.append(lambda x: (bicubic_upsample(x[0], H=self.bottom_width, W=self.bottom_width), x[1])) # transfer this part of code bicubic_upsample(x, H, W)
        self.specs.append(
            LayerSpec(
                PositionEmbedding,
                bottom_width=(self.bottom_width*2)**2, 
                embedding_dim=embed_dim
            )
        ) #self.specs.append(lambda x: x + self.pos_embed[1])
        self.specs.append(lambda x: (x[0], x[1]))
        # self.blocks_2 = StageBlock(
        #                 depth=depth[1],
        #                 dim=embed_dim, 
        #                 num_heads=num_heads, 
        #                 mlp_ratio=mlp_ratio, 
        #                 qkv_bias=qkv_bias, 
        #                 qk_scale=qk_scale,
        #                 drop=drop_rate, 
        #                 attn_drop=attn_drop_rate, 
        #                 drop_path=0, 
        #                 act_layer=act_layer,
        #                 norm_layer=norm_layer,
        #                 window_size=16
        #                 )
        for _ in range(depth[1]):
            self.specs.append(
                LayerSpec(ParallelTransformerBlock,
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, 
                        qk_scale=qk_scale,
                        drop=drop_rate, 
                        attn_drop=attn_drop_rate, 
                        drop_path=0, 
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        window_size=self.bottom_width*2,
                        init_method=self.init_method,
                        output_layer_init_method=self.output_layer_init_method
                    )
                )
        self.specs.append(lambda x: (bicubic_upsample(x[0], H=self.bottom_width*2, W=self.bottom_width*2), x[1]))
        self.specs.append(
            LayerSpec(
            PositionEmbedding,
            bottom_width=(self.bottom_width*4)**2, 
            embedding_dim=embed_dim
        )) #self.specs.append(lambda x: x + self.pos_embed[2])

        # self.blocks_3 = StageBlock(
        #                 depth=depth[2],
        #                 dim=embed_dim, 
        #                 num_heads=num_heads, 
        #                 mlp_ratio=mlp_ratio, 
        #                 qkv_bias=qkv_bias, 
        #                 qk_scale=qk_scale,
        #                 drop=drop_rate, 
        #                 attn_drop=attn_drop_rate, 
        #                 drop_path=0, 
        #                 act_layer=act_layer,
        #                 norm_layer=norm_layer,
        #                 window_size=32
        #                 )

        for _ in range(depth[2]):
            self.specs.append(
                LayerSpec(ParallelTransformerBlock,
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, 
                        qk_scale=qk_scale,
                        drop=drop_rate, 
                        attn_drop=attn_drop_rate, 
                        drop_path=0, 
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        window_size=self.bottom_width*4,
                        init_method=self.init_method,
                        output_layer_init_method=self.output_layer_init_method
                    )
                )
        self.specs.append(lambda x: (pixel_upsample(x[0], H=self.bottom_width*4, W=self.bottom_width*4), x[1]))
        self.specs.append(
            LayerSpec(
                PositionEmbedding,
                bottom_width=(self.bottom_width*8)**2, 
                embedding_dim=embed_dim//4
        ))    #self.specs.append(lambda x: x + self.pos_embed[3])

        self.specs.append(lambda x: (x[0].view(self.bs, self.bottom_width*8, self.bottom_width*8, embed_dim//4), x[1]))    # transform this part of code: x = x.view(B, H, W, C)
        self.specs.append(lambda x: (window_partition(x[0], self.window_size), x[1]))    # transform this part of code: x = window_partition(x, self.window_size)
        self.specs.append(lambda x: (x[0].view(-1, self.window_size*self.window_size, embed_dim//4),x[1])) # this seems to be wrong self.specs.append(lambda x: x.view(-1, 128*128, 256))    # transform this part of code: x = x.view(-1, self.window_size*self.window_size, C)        
        #self.specs.append(lambda x: print("PPPPPPPPPPPPPPPPPPPP lambda ***** x size: {}".format(x.size())))
        #self.specs.append(lambda x: x.view(1, x.size()[0], x.size()[1], x.size()[2]))
        self.specs.append(lambda x: (x[0], torch.zeros((100, 100),
                                    dtype=torch.bool,
                                    device=torch.cuda.current_device()))) # hongyi: this is a dirty hack

        # self.blocks_4 = StageBlock(
        #                 depth=depth[3],
        #                 dim=embed_dim//4, 
        #                 num_heads=num_heads, 
        #                 mlp_ratio=mlp_ratio, 
        #                 qkv_bias=qkv_bias, 
        #                 qk_scale=qk_scale,
        #                 drop=drop_rate, 
        #                 attn_drop=attn_drop_rate, 
        #                 drop_path=0, 
        #                 act_layer=act_layer,
        #                 norm_layer=norm_layer,
        #                 window_size=self.window_size
        #                 )
        for _ in range(depth[3]):
            self.specs.append(
                LayerSpec(ParallelTransformerBlock,
                        dim=embed_dim//4,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, 
                        qk_scale=qk_scale,
                        drop=drop_rate, 
                        attn_drop=attn_drop_rate, 
                        drop_path=0, 
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        window_size=self.window_size,
                        init_method=self.init_method,
                        output_layer_init_method=self.output_layer_init_method
                    )
                )
        self.specs.append(lambda x: (x[0].view(-1, self.window_size, self.window_size, embed_dim//4), x[1])) # transform this part of code: x = x.view(-1, self.window_size, self.window_size, C)
        self.specs.append(lambda x: (window_reverse(x[0], self.window_size, self.bottom_width*8, self.bottom_width*8).view(self.bs, self.bottom_width * 8 * self.bottom_width *8, embed_dim//4), x[1]))    # transform this part: x = window_reverse(x, self.window_size, H, W).view(B,H*W,C)

        self.specs.append(lambda x: (pixel_upsample(x[0], H=self.bottom_width*8, W=self.bottom_width*8), x[1]))
        self.specs.append(LayerSpec(
                PositionEmbedding,
                bottom_width=(self.bottom_width*16)**2, 
                embedding_dim=embed_dim//16
        )) #self.specs.append(lambda x: x + self.pos_embed[4])
        self.specs.append(lambda x: (x[0].view(self.bs, self.bottom_width*16, self.bottom_width*16, embed_dim//16), x[1])) # transform part of code: x = x.view(B, H, W, C)
        self.specs.append(lambda x: (window_partition(x[0], self.window_size),x[1]))
        self.specs.append(lambda x: (x[0].view(-1, self.window_size*self.window_size, embed_dim//16), x[1]))    # x = x.view(-1, self.window_size*self.window_size, C)
        

        # self.blocks_5 = StageBlock(
        #                 depth=depth[4],
        #                 dim=embed_dim//16, 
        #                 num_heads=num_heads, 
        #                 mlp_ratio=mlp_ratio, 
        #                 qkv_bias=qkv_bias, 
        #                 qk_scale=qk_scale,
        #                 drop=drop_rate, 
        #                 attn_drop=attn_drop_rate, 
        #                 drop_path=0, 
        #                 act_layer=act_layer,
        #                 norm_layer=norm_layer,
        #                 window_size=self.window_size
        #                )

        for _ in range(depth[4]):
            self.specs.append(
                LayerSpec(ParallelTransformerBlock,
                        dim=embed_dim//16,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, 
                        qk_scale=qk_scale,
                        drop=drop_rate, 
                        attn_drop=attn_drop_rate, 
                        drop_path=0, 
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        window_size=self.window_size,
                        init_method=self.init_method,
                        output_layer_init_method=self.output_layer_init_method
                    )
                )
        self.specs.append(lambda x: (x[0].view(-1, self.window_size, self.window_size, embed_dim//16),x[1]))
        self.specs.append(lambda x: (window_reverse(x[0], self.window_size, self.bottom_width*16, self.bottom_width*16).view(self.bs, self.bottom_width*16*self.bottom_width*16, embed_dim//16), x[1]))

        self.specs.append(lambda x: (pixel_upsample(x[0], H=self.bottom_width*16, W=self.bottom_width*16), x[1]))
        self.specs.append(LayerSpec(
            PositionEmbedding,
            bottom_width=(self.bottom_width*32)**2, 
            embedding_dim=embed_dim//64
        ))         #self.specs.append(lambda x: x + self.pos_embed[5])
        self.specs.append(lambda x: (x[0].view(self.bs, self.bottom_width*32, self.bottom_width*32, embed_dim//64), x[1]))
        self.specs.append(lambda x: (window_partition(x[0], self.window_size), x[1]))
        self.specs.append(lambda x: (x[0].view(-1, self.window_size*self.window_size, embed_dim//64), x[1]))
        

        # self.blocks_6 = StageBlock(
        #                 depth=depth[5],
        #                 dim=embed_dim//64, 
        #                 num_heads=num_heads, 
        #                 mlp_ratio=mlp_ratio, 
        #                 qkv_bias=qkv_bias, 
        #                 qk_scale=qk_scale,
        #                 drop=drop_rate, 
        #                 attn_drop=attn_drop_rate, 
        #                 drop_path=0, 
        #                 act_layer=act_layer,
        #                 norm_layer=norm_layer,
        #                 window_size=self.window_size
        #                 )

        #self.specs.append(lambda x: (x, torch.zeros((100, 100),dtype=torch.bool,device=torch.cuda.current_device()))) # hongyi: this is a dirty hack
        for _ in range(depth[5]):
            self.specs.append(
                LayerSpec(ParallelTransformerBlock,
                        dim=embed_dim//64,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, 
                        qk_scale=qk_scale,
                        drop=drop_rate, 
                        attn_drop=attn_drop_rate, 
                        drop_path=0, 
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        window_size=self.window_size,
                        init_method=self.init_method,
                        output_layer_init_method=self.output_layer_init_method
                    )
            )

        self.specs.append(lambda x: (x[0].view(-1, self.window_size, self.window_size, embed_dim//64), x[1]))
        self.specs.append(lambda x: window_reverse(x[0], self.window_size, self.bottom_width*32, self.bottom_width*32).view(self.bs,self.bottom_width*32,self.bottom_width*32,embed_dim//64).permute(0,3,1,2))
                                        

        # TODO(hwang): currently we do not seem to be able to handle this
        # see if we can do this later
        # for i in range(len(self.pos_embed)):
        #     trunc_normal_(self.pos_embed[i], std=.02)
        #self.deconv = nn.Sequential(
        #    nn.Conv2d(self.embed_dim//64, 3, 1, 1, 0)
        #)
        self.specs.append(LayerSpec(nn.Conv2d,
                    self.embed_dim//64,
                    3,
                    1,
                    1,
                    0
                ))
        #self.specs.append(lambda x: print("PPPPPPPPPPPPPPPPPPPP lambda ***** x size: {}".format(x.size())))
        interval = 0
        #self.specs.append(lambda x: self.deconv(x))
        super().__init__(layers=self.specs,
                         loss_fn=DummyGenLoss,
                         topology=topology,
                         activation_checkpoint_interval=interval,
                         partition_method='type:transformer')

    # def forward(self, z, epoch):
    #     if self.args.latent_norm:
    #         latent_size = z.size(-1)
    #         z = (z/z.norm(dim=-1, keepdim=True) * (latent_size ** 0.5))
    #     if self.args.latent_norm:
    #         latent_size = z.size(-1)
    #         z = (z/z.norm(dim=-1, keepdim=True) * (latent_size ** 0.5))
    #     #if self.l2_size == 0:
    #     #    x = self.l1(z).view(-1, self.bottom_width ** 2, self.embed_dim)
    #     #elif self.l2_size > 1000:
    #     #    x = self.l1(z).view(-1, self.bottom_width ** 2, self.l2_size//16)
    #     #    x = self.l2(x)
    #     #else:
    #     #    x = self.l1(z).view(-1, self.bottom_width ** 2, self.l2_size)
    #     #    x = self.l2(x)

    #     x = self.l1(z).view(-1, self.bottom_width ** 2, self.embed_dim)
            
    #     x = x + self.pos_embed[0]
    #     B = x.size()
    #     H, W = self.bottom_width, self.bottom_width
    #     x = self.blocks_1(x)
        
    #     x, H, W = bicubic_upsample(x, H, W)
    #     x = x + self.pos_embed[1]
    #     B, _, C = x.size()
    #     x = self.blocks_2(x)
        
    #     x, H, W = bicubic_upsample(x, H, W)
    #     x = x + self.pos_embed[2]
    #     B, _, C = x.size()
    #     x = self.blocks_3(x)
        
    #     x, H, W = pixel_upsample(x, H, W)
    #     x = x + self.pos_embed[3]
    #     B, _, C = x.size()
    #     x = x.view(B, H, W, C)
    #     x = window_partition(x, self.window_size)
    #     x = x.view(-1, self.window_size*self.window_size, C)
    #     x = self.blocks_4(x)

    #     x = x.view(-1, self.window_size, self.window_size, C)
    #     x = window_reverse(x, self.window_size, H, W).view(B,H*W,C)
            
    #     x, H, W = pixel_upsample(x, H, W)
    #     x = x + self.pos_embed[4]
    #     B, _, C = x.size()
    #     x = x.view(B, H, W, C) 
    #     x = window_partition(x, self.window_size) 
    #     x = x.view(-1, self.window_size*self.window_size, C) 
    #     x = self.blocks_5(x) 
    #     x = x.view(-1, self.window_size, self.window_size, C) 
    #     x = window_reverse(x, self.window_size, H, W).view(B,H*W,C) 
                                        
    #     x, H, W = pixel_upsample(x, H, W) 
    #     x = x + self.pos_embed[5]
    #     B, _, C = x.size()
    #     x = x.view(B, H, W, C) 
    #     x = window_partition(x, self.window_size) 
    #     x = x.view(-1, self.window_size*self.window_size, C) 
    #     x = self.blocks_6(x) 
    #     x = x.view(-1, self.window_size, self.window_size, C) 
    #     x = window_reverse(x, self.window_size, H, W).view(B,H,W,C).permute(0,3,1,2) ##     
        
    #     output = self.deconv(x)
    #     return output


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)


class DisBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=leakyrelu, norm_layer=nn.LayerNorm, separate=0):
        super().__init__()
        self.norm1 = CustomNorm(norm_layer, dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = CustomNorm(norm_layer, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gain = np.sqrt(0.5) if norm_layer == "none" else 1

    def forward(self, x):
        x = x*self.gain + self.drop_path(self.attn(self.norm1(x)))*self.gain
        x = x*self.gain + self.drop_path(self.mlp(self.norm2(x)))*self.gain
        return x


# class Discriminator(nn.Module):
#     def __init__(self, args, img_size=32, patch_size=None, in_chans=3, num_classes=1, embed_dim=None, depth=7,
#                  num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
#                  drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.num_classes = num_classes
#         self.num_features = embed_dim = self.embed_dim = args.df_dim  
        
#         depth = args.d_depth
#         self.args = args
#         self.patch_size = patch_size = args.patch_size
#         norm_layer = args.d_norm
#         act_layer = args.d_act
#         self.window_size = args.d_window_size
        
#         self.fRGB_1 = nn.Conv2d(3, embed_dim//8, kernel_size=patch_size, stride=patch_size, padding=0)
#         self.fRGB_2 = nn.Conv2d(3, embed_dim//8, kernel_size=patch_size, stride=patch_size, padding=0)
#         self.fRGB_3 = nn.Conv2d(3, embed_dim//4, kernel_size=patch_size, stride=patch_size, padding=0)
#         self.fRGB_4 = nn.Conv2d(3, embed_dim//2, kernel_size=patch_size, stride=patch_size, padding=0)
        
#         num_patches_1 = (args.img_size // patch_size)**2
#         num_patches_2 = ((args.img_size//2) // patch_size)**2
#         num_patches_3 = ((args.img_size//4) // patch_size)**2
#         num_patches_4 = ((args.img_size//8) // patch_size)**2

#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed_1 = nn.Parameter(torch.zeros(1, num_patches_1, embed_dim//8))
#         self.pos_embed_2 = nn.Parameter(torch.zeros(1, num_patches_2, embed_dim//4))
#         self.pos_embed_3 = nn.Parameter(torch.zeros(1, num_patches_3, embed_dim//2))
#         self.pos_embed_4 = nn.Parameter(torch.zeros(1, num_patches_4, embed_dim))
        
#         self.pos_drop = nn.Dropout(p=drop_rate)
        
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
#         self.blocks_1 = nn.ModuleList([
#             DisBlock(
#                 dim=embed_dim//8, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, act_layer=act_layer, norm_layer=norm_layer)
#             for i in range(depth+1)])
#         self.blocks_2 = nn.ModuleList([
#             DisBlock(
#                 dim=embed_dim//4, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, act_layer=act_layer, norm_layer=norm_layer)
#             for i in range(depth)])
#         self.blocks_3 = nn.ModuleList([
#             DisBlock(
#                 dim=embed_dim//2, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, act_layer=act_layer, norm_layer=norm_layer)
#             for i in range(depth)])
#         self.blocks_4 = nn.ModuleList([
#             DisBlock(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, act_layer=act_layer, norm_layer=norm_layer)
#             for i in range(depth)])
#         self.last_block = nn.Sequential(
# #             Block(
# #                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
# #                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer),
#             Block(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], act_layer=act_layer, norm_layer=norm_layer)
#             )
        
#         self.norm = CustomNorm(norm_layer, embed_dim)
#         self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

#         trunc_normal_(self.pos_embed_1, std=.02)
#         trunc_normal_(self.pos_embed_2, std=.02)
#         trunc_normal_(self.pos_embed_3, std=.02)
#         trunc_normal_(self.pos_embed_4, std=.02)
#         trunc_normal_(self.cls_token, std=.02)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

            
#     def forward_features(self, x):
#         if "None" not in self.args.diff_aug:
#             x = DiffAugment(x, self.args.diff_aug, True)
        
#         x_1 = self.fRGB_1(x).flatten(2).permute(0,2,1)
#         x_2 = self.fRGB_2(nn.AvgPool2d(2)(x)).flatten(2).permute(0,2,1)
#         x_3 = self.fRGB_3(nn.AvgPool2d(4)(x)).flatten(2).permute(0,2,1)
#         x_4 = self.fRGB_4(nn.AvgPool2d(8)(x)).flatten(2).permute(0,2,1)
#         B = x.shape[0]
        
#         x = x_1 + self.pos_embed_1
#         x = self.pos_drop(x)
#         H = W = self.args.img_size // self.patch_size
#         B, _, C = x.size()
#         x = x.view(B, H, W, C)
#         x = window_partition(x, self.window_size)
#         x = x.view(-1, self.window_size*self.window_size, C)
#         for blk in self.blocks_1:
#             x = blk(x)
#         x = x.view(-1, self.window_size, self.window_size, C)
#         x = window_reverse(x, self.window_size, H, W).view(B,H*W,C)
            
#         _, _, C = x.shape
#         x = x.permute(0, 2, 1).view(B, C, H, W)
# #         x = SpaceToDepth(2)(x)
#         x = nn.AvgPool2d(2)(x)
#         _, _, H, W = x.shape
#         x = x.flatten(2).permute(0, 2, 1)
#         x = torch.cat([x, x_2], dim=-1)
#         x = x + self.pos_embed_2
        
#         for blk in self.blocks_2:
#             x = blk(x)
        
#         _, _, C = x.shape
#         x = x.permute(0, 2, 1).view(B, C, H, W)
# #         x = SpaceToDepth(2)(x)
#         x = nn.AvgPool2d(2)(x)
#         _, _, H, W = x.shape
#         x = x.flatten(2).permute(0, 2, 1)
#         x = torch.cat([x, x_3], dim=-1)
#         x = x + self.pos_embed_3
        
#         for blk in self.blocks_3:
#             x = blk(x)
            
#         _, _, C = x.shape
#         x = x.permute(0, 2, 1).view(B, C, H, W)
# #         x = SpaceToDepth(2)(x)
#         x = nn.AvgPool2d(2)(x)
#         _, _, H, W = x.shape
#         x = x.flatten(2).permute(0, 2, 1)
#         x = torch.cat([x, x_4], dim=-1)
#         x = x + self.pos_embed_4
        
#         for blk in self.blocks_4:
#             x = blk(x)
            
#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = self.last_block(x)
#         x = self.norm(x)
#         return x[:,0]

#     def forward(self, x):
#         x = self.forward_features(x)
#         x = self.head(x)
#         return x
