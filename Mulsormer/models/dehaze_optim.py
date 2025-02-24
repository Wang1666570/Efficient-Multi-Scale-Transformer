# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from models.utilArch import AttBlock


##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans*patch_size**2, kernel_size=kernel_size, padding=kernel_size//2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size))

    def forward(self, x):
        x = self.proj(x)
        return x

##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=24, num_blocks=[2, 4, 4, 4], num_refinement_blocks=1,
                 ffn_scale=2, bias=False, LayerNorm_type='WithBias', dual_pixel_task=True):
        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        super(Restormer, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[
            AttBlock(dim=dim, ffn_scale=ffn_scale, bias=bias) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            AttBlock(dim=int(dim * 2 ** 1), ffn_scale=ffn_scale, bias=bias) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            AttBlock(dim=int(dim * 2 ** 2), ffn_scale=ffn_scale, bias=bias) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            AttBlock(dim=int(dim * 2 ** 3), ffn_scale=ffn_scale, bias=bias) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            AttBlock(dim=int(dim * 2 ** 2), ffn_scale=ffn_scale, bias=bias) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            AttBlock(dim=int(dim * 2 ** 1), ffn_scale=ffn_scale, bias=bias) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level1 = nn.Sequential(*[
            AttBlock(dim=int(dim * 2 ** 1), ffn_scale=ffn_scale, bias=bias) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            AttBlock(dim=int(dim * 2 ** 1), ffn_scale=ffn_scale, bias=bias) for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################
        self.output_first = nn.Conv2d(int(dim * 2 ** 1), out_channels+1, kernel_size=1, bias=bias)
        self.output_last = nn.Conv2d(int(dim * 2 ** 1), out_channels+1, kernel_size=1, stride=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        # print("inp_enc_level1: ", inp_enc_level1.shape)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        # print("out_enc_level1: ", out_enc_level1.shape)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        # print("inp_enc_level2: ", inp_enc_level2.shape)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        # print("out_enc_level2: ", out_enc_level2.shape)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        # print("inp_enc_level3: ", inp_enc_level3.shape)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        # print("out_enc_level3: ", out_enc_level3.shape)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        # print("in_enc_level4: ", inp_enc_level4.shape)
        latent = self.latent(inp_enc_level4)
        # print("latent: ", latent.shape)

        inp_dec_level3 = self.up4_3(latent)
        # print("inp_dec_level3: ", inp_dec_level3.shape)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        # print("inp_dec_level3: ", inp_dec_level3.shape)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        # print("inp_dec_level3: ", inp_dec_level3.shape)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        # print("out_dec_level3: ", out_dec_level3.shape)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        # print("inp_dec_level2: ", inp_dec_level2.shape)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        # print("inp_dec_level2: ", inp_dec_level2.shape)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        # print("inp_dec_level2: ", inp_dec_level2.shape)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        # print("out_dec_level2: ", out_dec_level2.shape)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        # print("inp_dec_level1: ", inp_dec_level1.shape)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        # print("inp_dec_level1: ", inp_dec_level1.shape)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        # print("out_dec_level1: ", out_dec_level1.shape)
        out_dec_first = self.output_first(out_dec_level1)
        K, B = torch.split(out_dec_first, (1, 3), dim=1)
        out_dec_first = K * inp_img - B + inp_img
        # print("out_dec_first: ", out_dec_first.shape)

        out_dec_level1 = self.refinement(out_dec_level1)
        # print("out_dec_level1: ", out_dec_level1.shape)

        out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
        out_dec_level1 = self.output_last(out_dec_level1)
        K, B = torch.split(out_dec_level1, (1, 3), dim=1)
        out_dec_level1 = K * inp_img - B + inp_img

        return [out_dec_first, out_dec_level1]



if __name__ == '__main__':

    input = torch.randn(2, 3, 256, 256)
    model = Restormer()
    out = model(input)
    print("out: ", out[-1].size())