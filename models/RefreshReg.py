import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Activate import Activate
from models.KPConv import block_decider
from models.dual_graph_convolution import DGC
from models.local_ppf import LocalInteractive
from transformer.modules.do_trans import Transformer


class RefreshReg(nn.Module):
    def __init__(self, config):
        super().__init__()
        r = config.first_subsampling_dl * config.conv_radius
        in_dim, out_dim = config.in_feats_dim, config.first_feats_dim

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skips = [] # record the index of layers to be needed in decoder layer.
        self.encoder_skip_dims = [] # record the dims before pooling or strided-conv.
        block_i, layer_ind = 0, 0
        for block in config.architecture:
            if 'upsample' in block:
                break
            if np.any([skip_block in block for skip_block in ['strided', 'pool']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)
            self.encoder_blocks.append(block_decider(block_name=block,
                                                     radius=r,
                                                     in_dim=in_dim,
                                                     out_dim=out_dim,
                                                     use_bn=config.use_batch_norm,
                                                     bn_momentum=config.batch_norm_momentum,
                                                     layer_ind=layer_ind,
                                                     config=config))

            in_dim = out_dim // 2 if 'simple' in block else out_dim
            if np.any([skip_block in block for skip_block in ['strided', 'pool']]):
                r *= 2
                out_dim *= 2
                layer_ind += 1
            block_i += 1

        self.bottleneck = nn.Conv1d(out_dim, config.gnn_feats_dim, 1)

        # Dual Graph Convolution block
        self.dgc = DGC(input_features_dim=256)

        # ppf_local feature
        self.ppf = LocalInteractive(feat_dims=config.gnn_feats_dim,
                                    gcn_k=config.dgcnn_k,
                                    ppf_k=config.ppf_k,
                                    radius=config.first_subsampling_dl * config.radius_mul,
                                    bottleneck=config.bottleneck,
                                    nhead=config.num_head)

        self.fused = nn.Sequential(
                nn.Conv1d(256 * 2, 256 * 2, 1),
                nn.InstanceNorm1d(256 * 2),
                nn.LeakyReLU(0.2),
                nn.Conv1d(256 * 2, 256, 1),
                nn.InstanceNorm1d(256),
                nn.LeakyReLU(0.2)
                )

        self.conv_down1 = nn.Conv1d(256, 256 // 8, 1, bias=False)
        self.conv_down2 = nn.Conv1d(256, 256 // 8, 1, bias=False)

        self.conv_up = nn.Conv1d(256 // 8, 256, 1)
        self.bn_up = nn.BatchNorm1d(256)

        self.activate = Activate()


        # Geometric Transformer block
        self.transformer = Transformer(
            config.input_dim,
            config.output_dim,
            config.hidden_dim,
            config.num_heads,
            config.blocks,
            config.sigma_d,
            config.sigma_a,
            config.angle_k,
            reduction_a=config.reduction_a,
        )
        self.pro_gnn = nn.Conv1d(config.gnn_feats_dim, config.gnn_feats_dim, 1)
        self.attn_score = nn.Conv1d(config.gnn_feats_dim, 1, 1)
        self.epsilon = nn.Parameter(torch.tensor(-5.0))

        # Decoder blocks
        out_dim = config.gnn_feats_dim + 2
        self.decoder_blocks = nn.ModuleList()
        self.decoder_skips = []
        layer = len(self.encoder_skip_dims) - 1

        self.decoder_blocks_m = nn.ModuleList()
        self.decoder_blocks_l = nn.ModuleList()
        cnt_upsample, mid_flag, low_flag = 0, True, True
        for block in config.architecture[block_i:]:
            if 'upsample' in block:
                layer_ind -= 1
                self.decoder_skips.append(block_i + 1)

            self.decoder_blocks.append(block_decider(block_name=block,
                                                     radius=r,
                                                     in_dim=in_dim,
                                                     out_dim=out_dim,
                                                     use_bn=config.use_batch_norm,
                                                     bn_momentum=config.batch_norm_momentum,
                                                     layer_ind=layer_ind,
                                                     config=config))

            if cnt_upsample >= 1:
                if cnt_upsample == 1 and mid_flag:
                    in_dim_clean = self.encoder_skip_dims[layer+1]
                    mid_flag = False
                else:
                    in_dim_clean = in_dim

                out_dim_clean = -1 if block == 'last_unary' else out_dim

                self.decoder_blocks_m.append(block_decider(block_name=block,
                                                           radius=r,
                                                           in_dim=in_dim_clean,
                                                           out_dim=out_dim_clean,
                                                           use_bn=config.use_batch_norm,
                                                           bn_momentum=config.batch_norm_momentum,
                                                           layer_ind=layer_ind,
                                                           config=config))

            if cnt_upsample >= 2:
                if cnt_upsample == 2 and low_flag:
                    in_dim_clean = self.encoder_skip_dims[layer+1]
                    low_flag = False
                else:
                    in_dim_clean = in_dim
                out_dim_clean = -1 if block == 'last_unary' else out_dim
                self.decoder_blocks_l.append(block_decider(block_name=block,
                                                           radius=r,
                                                           in_dim=in_dim_clean,
                                                           out_dim=out_dim_clean,
                                                           use_bn=config.use_batch_norm,
                                                           bn_momentum=config.batch_norm_momentum,
                                                           layer_ind=layer_ind,
                                                           config=config))

            in_dim = out_dim

            if 'upsample' in block:
                r *= 0.5
                in_dim += self.encoder_skip_dims[layer]
                layer -= 1
                out_dim = out_dim // 2
                cnt_upsample += 1

            block_i += 1

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        stack_points = inputs['points']
        stacked_normals = inputs['normals']
        stack_lengths = inputs['stacked_lengths']

        # 1. encoder
        batched_feats = inputs['feats']
        block_i = 0
        skip_feats = []
        for block in self.encoder_blocks:
            if block_i in self.encoder_skips:
                skip_feats.append(batched_feats)
            batched_feats = block(batched_feats, inputs)
            block_i += 1

        # 2.1 bottleneck layer
        batched_feats = self.bottleneck(batched_feats.transpose(0, 1).unsqueeze(0))

        len_src, len_tgt = stack_lengths[-1][0], stack_lengths[-1][1]
        coords_src, coords_tgt = stack_points[-1][:len_src], stack_points[-1][len_src:]
        coords_src, coords_tgt = coords_src.transpose(0, 1).unsqueeze(0), \
                                 coords_tgt.transpose(0, 1).unsqueeze(0) ##(1,3,n)

        normals_src = stacked_normals[-1][:len_src]
        normals_tgt = stacked_normals[-1][len_src:]
        normals_src = normals_src.transpose(0, 1).unsqueeze(0)
        normals_tgt = normals_tgt.transpose(0, 1).unsqueeze(0) ##(1,3,n)

        feats_src = batched_feats[:, :, :len_src]
        feats_tgt = batched_feats[:, :, len_src:]  #(1,256,n)

        # Pad the input tensor
        coords_src = pad_tensor(coords_src)
        coords_tgt = pad_tensor(coords_tgt)
        normals_src = pad_tensor(normals_src)
        normals_tgt = pad_tensor(normals_tgt)
        feats_src = pad_tensor(feats_src, target_dim=20)
        feats_tgt = pad_tensor(feats_tgt, target_dim=20)

        # 2.2.1 Dual Graph Convolution module
        feats_src_dgc = self.dgc(coords_src,feats_src,k=20)
        feats_tgt_dgc = self.dgc(coords_tgt,feats_tgt,k=20) # (1,256,n)

        # 2.2.2 ppf_local feature
        local_feats_src, local_feats_tgt = self.ppf(coords_src, feats_src, coords_tgt, feats_tgt, normals_src, normals_tgt)

        coords_src = coords_src.transpose(1, 2)
        coords_tgt = coords_tgt.transpose(1, 2)  ##(1,n,3)

        src_fused_feat = self.fused(torch.cat([feats_src_dgc,local_feats_src], dim=1))
        tgt_fused_feat = self.fused(torch.cat([feats_tgt_dgc,local_feats_tgt], dim=1))

        # 2.2.3 bilinear response module
        f_encoding_1 = F.relu(self.conv_down1(src_fused_feat))  # B,C/8,N
        f_encoding_2 = F.relu(self.conv_down2(src_fused_feat))

        f_encoding_channel = f_encoding_1.mean(dim=-1, keepdim=True)[0]  # B,C/8,1
        f_encoding_space = f_encoding_2.mean(dim=1, keepdim=True)[0]
        final_encoding = torch.matmul(f_encoding_channel, f_encoding_space)
        final_encoding = torch.sqrt(final_encoding + 1e-12)
        final_encoding = final_encoding + f_encoding_1 + f_encoding_2
        final_encoding = F.relu(self.bn_up(self.conv_up(final_encoding)))

        f_src_fused_out = src_fused_feat - final_encoding

        f_encoding_1 = F.relu(self.conv_down1(tgt_fused_feat))
        f_encoding_2 = F.relu(self.conv_down2(tgt_fused_feat))

        f_encoding_channel = f_encoding_1.mean(dim=-1, keepdim=True)[0]  # B,C/8,1
        f_encoding_space = f_encoding_2.mean(dim=1, keepdim=True)[0]
        final_encoding = torch.matmul(f_encoding_channel, f_encoding_space)
        final_encoding = torch.sqrt(final_encoding + 1e-12)
        final_encoding = final_encoding + f_encoding_1 + f_encoding_2
        final_encoding = F.relu(self.bn_up(self.conv_up(final_encoding)))  # B,C,N

        f_tgt_fused_out = tgt_fused_feat - final_encoding

        f_src_fused_out = self.activate(f_src_fused_out)
        f_tgt_fused_out = self.activate(f_tgt_fused_out)

        f_src_fused_out = f_src_fused_out.transpose(1, 2) #(1,n,256)
        f_tgt_fused_out = f_tgt_fused_out.transpose(1, 2)

        # 2.3 Transformer
        feats_tgt, feats_src = self.transformer(
            coords_tgt,
            coords_src,
            f_tgt_fused_out,
            f_src_fused_out,
        )  ##(1,n,256)

        feats_src = feats_src.transpose(1, 2)
        feats_tgt = feats_tgt.transpose(1, 2)

        batched_feats = torch.cat([feats_src, feats_tgt], dim=-1)
        batched_feats = self.pro_gnn(batched_feats)

        # 2.4 overlap score
        attn_scores_tmp = self.attn_score(batched_feats)
        attn_scores = attn_scores_tmp.squeeze(0).transpose(0, 1) # (n, 1)
        temperature = torch.exp(self.epsilon) + 0.03
        batched_feats_norm = batched_feats / (torch.norm(batched_feats, dim=1, keepdim=True) + 1e-8)
        batched_feats_norm = batched_feats_norm.squeeze(0).transpose(0, 1) # (n, c)
        feats_norm_src, feats_norm_tgt = batched_feats_norm[:len_src], \
                                         batched_feats_norm[len_src:]
        inner_product = torch.matmul(feats_norm_src, feats_norm_tgt.transpose(0, 1)) # (n1, n2), n1 + n2
        attn_scores_src, attn_scores_tgt = attn_scores[:len_src], attn_scores[len_src:]
        ol_scores_src = torch.matmul(torch.softmax(inner_product / temperature, dim=1), attn_scores_tgt) # (n1, 1)
        ol_scores_tgt = torch.matmul(torch.softmax(inner_product.transpose(0, 1) / temperature, dim=1), attn_scores_src) # (n2, 1)
        ol_scores = torch.cat([ol_scores_src, ol_scores_tgt], dim=0) # (n, 1)

        # 2.5 feats
        batched_feats_raw = batched_feats.squeeze(0).transpose(0, 1)  # (n, c)
        batched_feats = torch.cat([batched_feats_raw, attn_scores, ol_scores], dim=1)

        # 3. decoder
        cnt_decoder = 0
        for ind, block in enumerate(self.decoder_blocks):
            if block_i in self.decoder_skips:
                cnt_decoder += 1
                cur_skip_feats = skip_feats.pop()
                batched_feats = torch.cat([batched_feats, cur_skip_feats], dim=-1)
                if cnt_decoder >= 1:
                    if cnt_decoder == 1:
                        batched_feats_m = cur_skip_feats
                    else:
                        batched_feats_m = torch.cat([batched_feats_m, cur_skip_feats], dim=-1)
                if cnt_decoder >= 2:
                    if cnt_decoder == 2:
                        batched_feats_l = cur_skip_feats
                    else:
                        batched_feats_l = torch.cat([batched_feats_l, cur_skip_feats], dim=-1)

            if cnt_decoder >= 1:
                block_m = self.decoder_blocks_m[ind - 1]
                batched_feats_m = block_m(batched_feats_m, inputs)

            if cnt_decoder >= 2:
                block_l = self.decoder_blocks_l[ind - (self.decoder_skips[1] - self.decoder_skips[0] + 1)]
                batched_feats_l = block_l(batched_feats_l, inputs)

            batched_feats = block(batched_feats, inputs)
            block_i += 1

        overlap_scores = self.sigmoid(batched_feats[:, -2:-1])
        saliency_scores = self.sigmoid(batched_feats[:, -1:])
        batched_feats = batched_feats[:, :-2] / torch.norm(batched_feats[:, :-2], dim=1, keepdim=True)
        batched_feats_m = batched_feats_m / torch.norm(batched_feats_m, dim=1, keepdim=True)
        batched_feats_l = batched_feats_l / torch.norm(batched_feats_l, dim=1, keepdim=True)
        batched_feats = torch.cat([batched_feats, overlap_scores, saliency_scores], dim=-1)

        return batched_feats, batched_feats_m, batched_feats_l


def pad_tensor(input_tensor, target_dim=20, pad_value=0):
    """
    Pad the input tensor along the 2nd dimension to reach the target size.

    Args:
        input_tensor (torch.Tensor): Input tensor with shape (batch_size, feature_dim, seq_len).
        target_dim (int): Target dimension size for padding, default is 20.
        pad_value (float): Value used for padding, default is 0.

    Returns:
        torch.Tensor: Padded tensor with shape (batch_size, feature_dim, target_dim) if original seq_len is smaller than target_dim,
                      otherwise returns the original tensor.
    """
    if input_tensor.shape[2] < target_dim:
        padding_size = target_dim - input_tensor.shape[2]
        batch_size, feature_dim = input_tensor.shape[:2]

        padding = torch.full((batch_size, feature_dim, padding_size),
                             pad_value,
                             dtype=input_tensor.dtype,
                             device=input_tensor.device)

        return torch.cat([input_tensor, padding], dim=2)
    else:
        return input_tensor