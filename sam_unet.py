import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=256, patch_size=32, in_channels=3, embed_dim=1280):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class HyperAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels * 4, in_channels)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(2, 0, 1)
        x_norm = self.norm(x_flat)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x_att = x_flat + self.dropout(attn_output)
        x_mlp = self.mlp(x_att)
        x_out = x_att + self.dropout(x_mlp)
        x_out = x_out.permute(1, 2, 0).reshape(B, C, H, W)
        return x_out


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True, dropout_rate=0.1):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x, skip):
        x = self.up(x)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class FusionBlock(nn.Module):
    def __init__(self, in_channels_res, in_channels_sam, out_channels, dropout_rate=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels_res + in_channels_sam, out_channels, kernel_size=4, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x_res, x_sam):
        if x_sam.shape[2:] != x_res.shape[2:]:
            x_sam = F.interpolate(x_sam, size=x_res.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x_res, x_sam], dim=1)
        return self.conv(x)


class SAM_U_Net_Model(nn.Module):
    def __init__(self, out_channels, sam_model_registry, checkpoint_path, use_patch_embed=True, bilinear=True, pretrained=True, dropout_rate=0.1):
        super().__init__()
        self.use_patch_embed = use_patch_embed

        resnet = models.resnet101(pretrained=pretrained)
        self.input_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        sam_encoder_full = sam.image_encoder
        for param in sam_encoder_full.parameters():
            param.requires_grad = False

        if self.use_patch_embed:
            self.sam_patch_embed = nn.Conv2d(3, 1280, kernel_size=16, stride=16)
            self.patch_dropout = nn.Dropout(dropout_rate)
            self.sam_blocks = nn.ModuleList(list(sam_encoder_full.blocks)[:6])
            self.sam_fusion = nn.Conv2d(1280 * 2, 1280, kernel_size=1)
        else:
            self.sam_encoder = nn.Sequential(*list(sam_encoder_full.blocks)[:6])

        self.fusion = nn.Sequential(
            nn.Conv2d(2048 + 1280, 2048, kernel_size=4, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        self.hyper_attention = nn.Sequential(
            nn.LayerNorm(2048),
            nn.MultiheadAttention(embed_dim=2048, num_heads=8, dropout=dropout_rate)
        )

        self.up1 = UpBlock(2048, 1024, 1024, bilinear, dropout_rate)
        self.up2 = UpBlock(1024, 512, 512, bilinear, dropout_rate)
        self.up3 = UpBlock(512, 256, 256, bilinear, dropout_rate)
        self.up4 = UpBlock(256, 64, 64, bilinear, dropout_rate)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, sam_input, unet_input):
        x0 = self.input_conv(unet_input)
        x1 = self.maxpool(x0)
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        if self.use_patch_embed:
            x_sam = self.sam_patch_embed(sam_input)
            x_sam = self.patch_dropout(x_sam)
            x_sam = x_sam.permute(0, 2, 3, 1)
            outputs = [block(x_sam) for block in self.sam_blocks]
            groupA = outputs[0] + outputs[2] + outputs[4]
            groupB = outputs[1] + outputs[3] + outputs[5]
            fused = torch.cat([groupA, groupB], dim=-1).permute(0, 3, 1, 2)
            x_sam = self.sam_fusion(fused)
        else:
            x_sam = self.sam_encoder(sam_input)

        x_fused = torch.cat([x4, F.interpolate(x_sam, size=x4.shape[2:], mode='bilinear', align_corners=True)], dim=1)
        x_fused = self.fusion(x_fused)

        B, C, H, W = x_fused.shape
        x_flat = x_fused.flatten(2).transpose(1, 2)
        normed = self.hyper_attention[0](x_flat)
        attn_output, _ = self.hyper_attention[1](normed.transpose(0, 1), normed.transpose(0, 1), normed.transpose(0, 1))
        x_flat = x_flat + attn_output.transpose(0, 1)
        x_fused = x_flat.transpose(1, 2).reshape(B, C, H, W)

        d1 = self.up1(x_fused, x3)
        d2 = self.up2(d1, x2)
        d3 = self.up3(d2, x1)
        d4 = self.up4(d3, x0)
        d5 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.out_conv(d5)
        return out
