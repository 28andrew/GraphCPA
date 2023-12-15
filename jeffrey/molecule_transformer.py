# Following Transformer-M, using same module parameter names so we can re-use a pretrained checkpoint that we previously created

import torch
from fairseq.models import FairseqEncoder
from fairseq.modules import LayerNorm, FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import nn
from torch.nn import GELU
from torch.nn.functional import multi_head_attention_forward


class MoleculeTransformer(FairseqEncoder):
    def __init__(self):
        super().__init__(dictionary=None)

        self.molecule_encoder = MoleculeEncoder()

        self.lm_head_transform_weight = nn.Linear(768, 768)
        self.activation_fn = GELU()
        self.layer_norm = LayerNorm(768)

    def forward(self, data, **extra):
        states, atom_output = self.molecule_encoder(data, None, None)
        # Get final transformer state as embedding
        x = states[-1].transpose(0, 1)
        return x, atom_output, None


class MoleculeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_module = FairseqDropout(0.1, module_name='TransformerMEncoderQM9')
        self.mode_prob = [0.2, 0.2, 0.6]
        self.atom_feature = AtomFeature()
        self.molecule_attn_bias = MoleculeAttnBias()
        self.emb_layer_norm = LayerNorm(768, export=False)

        self.layers = nn.ModuleList([])
        for _ in range(12):
            self.layers.append(EncoderLayer())

    def forward(self, data):
        x = data["x"]
        n_mol, n_atom = x.size()[:2]
        padding_mask = (x[:, :, 0]).eq(0)
        padding_mask_cls = torch.zeros(n_mol, 1, device=padding_mask.device, dtype=padding_mask.dtype)
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
        mask_2d = None
        x = self.atom_feature(data, mask_2d=mask_2d)
        bias = self.molecule_attn_bias(data, mask_2d=mask_2d)
        x = self.emb_layer_norm(x)
        x = self.dropout_module(x)
        x = x.transpose(0, 1)
        inner_states = [x]
        for layer in self.layers:
            x, _ = layer(x, self_attn_padding_mask=padding_mask, self_attn_mask=None, self_attn_bias=bias)

        return torch.stack(inner_states), None


class AtomFeature(nn.Module):
    def __init__(self):
        super(AtomFeature, self).__init__()
        self.atom_encoder = nn.Embedding(4609, 768, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(512, 768, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(512, 768, padding_idx=0)
        self.graph_token = nn.Embedding(1, 768)

    def forward(self, x, mask_2d=None):
        x, in_degree, out_degree = x['x'], x['in_degree'], x['out_degree']
        batch_size, nodes = x.size()[:2]
        node_feature = self.atom_encoder(x).sum(dim=-2)
        degree_feature = self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)
        if mask_2d:
            degree_feature = degree_feature * mask_2d[:, None, None]
        node_feature = node_feature + degree_feature
        token_feature = self.graph_token.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        return torch.cat([token_feature, node_feature], dim=1)


class MoleculeAttnBias(nn.Module):
    def __init__(self):
        super(MoleculeAttnBias, self).__init__()
        self.edge_encoder = nn.Embedding(1537, 32, padding_idx=0)
        self.edge_dis_encoder = nn.Embedding(131072, 1)
        self.spatial_pos_encoder = nn.Embedding(512, 32, padding_idx=0)
        self.graph_token_virtual_distance = nn.Embedding(1, 32)

    def forward(self, data, mask_2d=None):
        attn_bias, spatial_pos, x = data['attn_bias'], data['spatial_pos'], data['x']
        edge_input, attn_edge_type, = data['edge_input'], data['attn_edge_type']

        # THESE CALCULATIONS AS-IS FROM TRANSFORMER-M
        batch_size, nodes = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        spatial_pos_bias = spatial_pos_bias * mask_2d[:, None, None, None]

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias
        token = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + token
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + token
        spatial_pos_ = spatial_pos.clone()
        spatial_pos_[spatial_pos_ == 0] = 1
        spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
        if self.multi_hop_max_dist > 0:
            spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
            edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]

        edge_input = self.edge_encoder(edge_input).mean(-2)
        max_dist = edge_input.size(-2)
        edge_input_flat = edge_input.permute(
            3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
        edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(
            -1, self.num_heads, self.num_heads)[:max_dist, :, :])
        edge_input = edge_input_flat.reshape(
            max_dist, batch_size, nodes, nodes, self.num_heads).permute(1, 2, 3, 0, 4)
        edge_input = (edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
        edge_input = edge_input * mask_2d[:, None, None, None]
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        return graph_attn_bias + attn_bias.unsqueeze(1)


class EncoderLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout_module = CustomDropout(0.1)
        self.activation_dropout_module = FairseqDropout(0.1, module_name='EncoderLayer')
        self.activation_fn = GELU()
        self.self_attn = MultiheadAttention()
        self.self_attn_layer_norm = LayerNorm(768, export=False)
        self.fc1 = quant_noise(nn.Linear(768, 768), 0, 8)
        self.fc2 = quant_noise(nn.Linear(768, 768), 0, 8)
        self.final_layer_norm = LayerNorm(768, False)

    def forward(self, x):
        # Create residual connection
        residual = x
        x, attn = self.self_attn(x, x, x, None, None, False, None)
        x = self.dropout_module(x)
        x = residual + x

        # Another residual connection
        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.activation_dropout_module(x)
        x = residual + x
        return x, attn


# 'Attention is all you need' multihead attention
class MultiheadAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout_module = FairseqDropout(0.1, module_name='MultiheadAttention')
        self.k_proj = quant_noise(nn.Linear(768, 768, bias=True), 0, 8)
        self.v_proj = quant_noise(nn.Linear(768, 768, bias=True), 0, 8)
        self.q_proj = quant_noise(nn.Linear(768, 768, bias=True), 0, 8)
        self.out_proj = quant_noise(nn.Linear(768, 768, bias=True), 0, 8)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None):
        return multi_head_attention_forward(
            query,
            key,
            value,
            768,
            8,
            torch.empty([0]),
            torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
            None,
            None,
            False,
            0.1,
            self.out_proj.weight,
            self.out_proj.bias,
            self.training,
            key_padding_mask,
            need_weights,
            attn_mask,
            use_separate_proj_weight=True,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight
        )


class CustomDropout(nn.Module):
    def __init__(self, probability, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probability = probability

    def forward(self, x):
        keep_prob = 1 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device).floor_()
        return x.div(keep_prob) * random_tensor


class ClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 768)
        self.activation_fn = GELU()
        self.out_proj = nn.Linear(768, 1)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.out_proj(x)
        return x
