import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder
from .transformer import FeatureTransformer, FeatureFlowAttention
from .matching import global_correlation_softmax, local_correlation_softmax
from .geometry import flow_warp
from .utils import normalize_img, feature_add_position


class GMFlow(nn.Module):
    def __init__(self,
                 num_scales=1,
                 upsample_factor=8,
                 feature_channels=128,
                 attention_type='swin',
                 num_transformer_layers=6,
                 ffn_dim_expansion=4,
                 num_head=1,
                 backbone = None,
                 **kwargs,
                 ):
        super(GMFlow, self).__init__()

        self.num_scales = num_scales
        self.feature_channels = feature_channels
        self.upsample_factor = upsample_factor
        self.attention_type = attention_type
        self.num_transformer_layers = num_transformer_layers

        # ViT backbone
        self.backbone = backbone

        # flow propagation with self-attn
        self.feature_flow_attn = FeatureFlowAttention(in_channels=feature_channels)

        # convex upsampling: concat feature0 and flow as input
        self.upsampler = nn.Sequential(nn.Conv2d(2 + feature_channels, 256, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))

    def extract_feature(self, img0, img1):
        concat = torch.stack((img0, img1), dim=1).permute(0, 1, 3, 4, 2)  # [B, T, H, W, C]
        features = self.backbone(concat)  # [B, T, H, W, d]
        chunks = torch.chunk(features, chunks = 2, dim = 1)

        feature0, feature1 = [], []
        feature0.append(chunks[0])
        feature1.append(chunks[1])

        return feature0, feature1

    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=8,
                      ):
        if bilinear:
            up_flow = F.interpolate(flow, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * upsample_factor

        else:
            # convex upsampling
            concat = torch.cat((flow, feature), dim=1)

            mask = self.upsampler(concat)
            b, flow_channel, h, w = flow.shape
            mask = mask.view(b, 1, 9, self.upsample_factor, self.upsample_factor, h, w)  # [B, 1, 9, K, K, H, W]
            mask = torch.softmax(mask, dim=2)

            up_flow = F.unfold(self.upsample_factor * flow, [3, 3], padding=1)
            up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)  # [B, 2, 9, 1, 1, H, W]

            up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 2, K, K, H, W]
            up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
            up_flow = up_flow.reshape(b, flow_channel, self.upsample_factor * h,
                                      self.upsample_factor * w)  # [B, 2, K*H, K*W]

        return up_flow

    def forward_image(self, img0, img1,
                attn_splits_list=None,
                corr_radius_list=None,
                prop_radius_list=None,
                pred_bidir_flow=False,
                **kwargs,
                ):

        results_dict = {}
        flow_preds = []

        #img0, img1 = normalize_img(img0, img1)  # [B, 3, H, W]

        # resolution low to high
        feature0_list, feature1_list = self.extract_feature(img0, img1)  # list of features

        flow = None

        assert len(attn_splits_list) == len(corr_radius_list) == len(prop_radius_list) == self.num_scales

        for scale_idx in range(self.num_scales):
            #第k个scale的feature
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]

            if pred_bidir_flow and scale_idx > 0:
                # predicting bidirectional flow with refinement
                feature0, feature1 = torch.cat((feature0, feature1), dim=0), torch.cat((feature1, feature0), dim=0)
                #直接当成颠倒img1和img2

            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))

            if scale_idx > 0:
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2
                #flow：上一次的整体flow预测

            if flow is not None:
                flow = flow.detach() #stop grad
                feature1 = flow_warp(feature1, flow)  # [B, C, H, W]
                #根据flow去lookup feature

            attn_splits = attn_splits_list[scale_idx]
            #第几次refine的windows attn的window数目
            corr_radius = corr_radius_list[scale_idx]
            #第几次refine的matching半径
            prop_radius = prop_radius_list[scale_idx]
            #第几次refine的光滑化半径

            # correlation and softmax
            if corr_radius == -1:  # global matching
                flow_pred = global_correlation_softmax(feature0, feature1, pred_bidir_flow)[0]
            else:  # local matching
                flow_pred = local_correlation_softmax(feature0, feature1, corr_radius)[0]

            # flow or residual flow
            flow = flow + flow_pred if flow is not None else flow_pred
            #除了第一次，都是残差flow，flow_pred提供梯度

            # upsample to the original resolution for supervison
            # smooth前upsampling
            if self.training and scale_idx < self.num_scales - 1:  # only need to upsample intermediate flow predictions at training time
                flow_bilinear = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor)
                flow_preds.append(flow_bilinear)
                #上采样到原始flow同大小
            if scale_idx == self.num_scales - 1:  # only need to upsample intermediate flow predictions at training time
                flow_convex = self.upsample_flow(flow, feature0)
                flow_preds.append(flow_convex)

            # flow propagation with self-attn
            if pred_bidir_flow and scale_idx == 0:
                feature0 = torch.cat((feature0, feature1), dim=0)  # [2*B, C, H, W] for propagation
            flow = self.feature_flow_attn(feature0, flow.detach(),
                                          local_window_attn=prop_radius > 0,
                                          local_window_radius=prop_radius)
            # smooth后upsampling

            # bilinear upsampling at training time except the last one
            if self.training and scale_idx < self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0, bilinear=True, upsample_factor=upsample_factor)
                flow_preds.append(flow_up)

            if scale_idx == self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0)
                flow_preds.append(flow_up)

        results_dict.update({'flow_preds': flow_preds})

        return results_dict