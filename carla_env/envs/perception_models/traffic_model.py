import sys
sys.path.append('.')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from functools import partial

import math
from torch.nn import Parameter


def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes,
                                        sampling_locations, attention_weights):


    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ =\
        sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                             dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):


        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)


        sampling_grid_l_ = sampling_grids[:, :, :,
                                          level].transpose(1, 2).flatten(0, 1)

        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)


    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(bs, num_heads * embed_dims,
                                              num_queries)
    return output.transpose(1, 2).contiguous()


class MultiArcFace(nn.Module):

    def __init__(self, in_features, out_features, n, s=20.0, m=0.5, easy_margin=False):
        super(MultiArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cluster_centres = n
        self.out_class_num = out_features // n
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        if label is None:

            input = input.view(input.size(0), -1)

            if input.dim() == 1:
                input = input.unsqueeze(0)

            B, C = input.shape
            return F.linear(F.normalize(input), F.normalize(self.weight))


        input = input.squeeze(1)
        input = input.squeeze(2)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        B,_ = cosine.shape
        cosine = cosine.view(B, self.cluster_centres, self.out_class_num)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = label
        one_hot = one_hot.unsqueeze(1)
        one_hot = one_hot.repeat(1,self.cluster_centres,1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder,self).__init__()
        self.config = config
        self.mul_arcface = MultiArcFace(self.config["mlp_out_channel"], self.config["n"] * self.config["out_class_num"], self.config["n"], s=20, m=0.5, easy_margin=True)


    def forward(self, agent_all_feature, lable=None):
        B, K, _,_ = agent_all_feature.shape
        prob = self.mul_arcface(agent_all_feature,lable)
        prob = prob.view(B, self.config['n'], self.config['out_class_num'], 1)
        res, idx = torch.max(prob, dim=1)
        return res.squeeze(-1)


class sca_light(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads   = config["num_heads"]
        self.num_pts     = config["num_sam_pts"]
        self.num_levels  = config["num_levels"]
        self.embed_dim   = config["embed_dim"]

        self.sampling_offset = nn.Linear(self.embed_dim,
                                         self.num_heads * self.num_pts * self.num_levels * 2)
        self.attention_weights = nn.Linear(self.embed_dim,
                                           self.num_heads * self.num_pts * self.num_levels)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(0.1)

        self.use_learned_ref = True
        if self.use_learned_ref:
            nq = config["num_query"]
            self.ref_points = nn.Parameter(torch.rand(1, nq, 2))
        else:
            self.ref_points = None

    def forward(self, query, single_feat, ref_2d, h, w):

        B, Lq, C = query.shape
        H = self.num_heads
        L = self.num_levels
        P = self.num_pts


        sampling_offsets = self.sampling_offset(query).relu()
        sampling_offsets = sampling_offsets.view(B, Lq, H, L, P, 2)

        attention_weights = self.attention_weights(query).relu()
        attention_weights = attention_weights.view(B, Lq, H, L, P)
        attention_weights = attention_weights.softmax(-1)

        spatial_shapes = torch.tensor([[h, w]], device=query.device)
        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

        if self.use_learned_ref:
            base = self.ref_points.expand(B, Lq, 2).unsqueeze(2).unsqueeze(3).unsqueeze(4)
            ref = base.expand(-1, -1, 1, L, 1, -1)
        else:
            ref = ref_2d

        sampling_locations = ref + sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        value = single_feat.view(B, h * w, H, -1)

        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )

        output = self.norm(self.dropout(output) + query)

        return output

class tsa(nn.Module):
    def __init__(self, config):
        super(tsa,self).__init__()
        self.config = config
        self.num_heads = self.config["num_heads"]
        self.embed_dim = self.config["embed_dim"]
        self.temporal_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, query, prev_embed=None):
        bypass = query
        if prev_embed is None:
            prev_embed = query

        output, _ = self.temporal_attn(query, prev_embed, prev_embed)
        output = self.norm(output + bypass)
        return output


class Encoder_lightformer(nn.Module):

    def __init__(self, config):
        super(Encoder_lightformer, self).__init__()
        self.embed_dim = config["embed_dim"]
        self.config = config
        self.num_query = self.config["num_query"]
        self.num_heads = self.config["num_heads"]
        self.tsa = tsa(self.config)
        self.sca = sca_light(self.config)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self.prev_embed = None


    def forward(self, query, all_img_feats):
        bs, _, _, h, w = all_img_feats.shape
        ref_2d = self.get_reference_points(h, w, bs)
        query = query.unsqueeze(0).repeat(bs, 1, 1)

        all_feats = all_img_feats.flatten(3).permute(0,1,3,2)
        _,num_imgs,resolu,_ = all_feats.shape
        output = None

        for i in range(num_imgs):
            single_feat = all_feats[:, i, :, :].view(bs, resolu, self.num_heads, -1)
            output = self.tsa(query, self.prev_embed)
            output = self.sca(output, single_feat, ref_2d, h, w)
            output = self.mlp(output) + output
            output = self.norm(output)
            self.prev_embed = output

        self.prev_embed = None

        return output

    def get_reference_points(self, H=4, W=11, bs=8, device='cuda'):
        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=torch.float, device=device), torch.linspace(0.5, W - 0.5, W, dtype=torch.float, device=device))
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
        return ref_2d


def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    return x

class LightFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.config["embed_dim"] = 256
        self.Light_img_encoder = models.resnet18(pretrained=True)
        self.Light_img_encoder.fc = None
        self.Light_img_encoder.avgpool = None
        self.Light_img_encoder.forward = partial(forward, self.Light_img_encoder)


        self.down_conv = nn.Sequential(
            nn.Conv2d(512, 1024,3),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 256,3),
            nn.BatchNorm2d(256)
        )
        self.mlp = nn.Sequential(
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,1024),
        )
        self.head1 = nn.Sequential(
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,1024)
        )

        self.traffic_light_distance_head = nn.Sequential(
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,22),
        )
        self.num_query = self.config["num_query"]
        self.query_embed = nn.Embedding(self.num_query, 256)
        self.encoder = Encoder_lightformer(self.config)
        self.Light_decoder = Decoder(self.config)

    def forward(self, images, traffic_light_label=None):

        image_num = self.config['image_num']
        B,_,c,h,w = images.shape
        images = images.reshape(B*image_num,c,h,w)
        vectors = self.Light_img_encoder(images)
        vectors = self.down_conv(vectors)


        _,c,h,w = vectors.shape
        vectors = vectors.view(B, image_num, c, h, w)

        query = self.query_embed.weight

        agent_all_feature = self.encoder(query, vectors)
        agent_all_feature = self.mlp(agent_all_feature)


        light_token = agent_all_feature[:, 0, :].unsqueeze(1)
        dist_token  = agent_all_feature[:, 1, :]

        head1_out = self.head1(light_token)
        head1_out = head1_out.unsqueeze(3)


        traffic_light_distance = self.traffic_light_distance_head(dist_token)

        Light = self.Light_decoder(head1_out, traffic_light_label)

        return Light,traffic_light_distance,agent_all_feature
