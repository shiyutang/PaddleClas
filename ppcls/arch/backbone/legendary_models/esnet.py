# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function
import math
import paddle
from paddle import ParamAttr, reshape, transpose, concat, split
import paddle.nn as nn
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D
from paddle.nn.initializer import KaimingNormal
from paddle.regularizer import L2Decay
import paddle.nn.functional as F


from ..base.theseus_layer import TheseusLayer
from ....utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {
    "ESNet_x0_25":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ESNet_x0_25_pretrained.pdparams",
    "ESNet_x0_5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ESNet_x0_5_pretrained.pdparams",
    "ESNet_x0_75":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ESNet_x0_75_pretrained.pdparams",
    "ESNet_x1_0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ESNet_x1_0_pretrained.pdparams",
}

MODEL_STAGES_PATTERN = {"ESNet": ["blocks[2]", "blocks[9]", "blocks[12]"]}

__all__ = list(MODEL_URLS.keys())


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.shape[0:4]
    channels_per_group = num_channels // groups
    x = reshape(
        x=x, shape=[batch_size, groups, channels_per_group, height, width])
    x = transpose(x=x, perm=[0, 2, 1, 3, 4])
    x = reshape(x=x, shape=[batch_size, num_channels, height, width])
    return x


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(TheseusLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 if_act=True):
        super().__init__()
        self.conv = Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
            bias_attr=False)

        self.bn = BatchNorm(
            out_channels,
            param_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.if_act = if_act
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            x = self.hardswish(x)
        return x


class SEModule(TheseusLayer):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2D(1)
        self.conv1 = Conv2D(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = Conv2D(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = paddle.multiply(x=identity, y=x)
        return x


class ESBlock1(TheseusLayer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pw_1_1 = ConvBNLayer(
            in_channels=in_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1)
        self.dw_1 = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=1,
            groups=out_channels // 2,
            if_act=False)
        self.se = SEModule(out_channels)

        self.pw_1_2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1)

    def forward(self, x):
        x1, x2 = split(
            x, num_or_sections=[x.shape[1] // 2, x.shape[1] // 2], axis=1)
        x2 = self.pw_1_1(x2)
        x3 = self.dw_1(x2)
        x3 = concat([x2, x3], axis=1)
        x3 = self.se(x3)
        x3 = self.pw_1_2(x3)
        x = concat([x1, x3], axis=1)
        return channel_shuffle(x, 2)


class ESBlock2(TheseusLayer):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # branch1
        self.dw_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            groups=in_channels,
            if_act=False)
        self.pw_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1)
        # branch2
        self.pw_2_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1)
        self.dw_2 = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=2,
            groups=out_channels // 2,
            if_act=False)
        self.se = SEModule(out_channels // 2)
        self.pw_2_2 = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1)
        self.concat_dw = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            groups=out_channels)
        self.concat_pw = ConvBNLayer(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.dw_1(x)
        x1 = self.pw_1(x1)
        x2 = self.pw_2_1(x)
        x2 = self.dw_2(x2)
        x2 = self.se(x2)
        x2 = self.pw_2_2(x2)
        x = concat([x1, x2], axis=1)
        x = self.concat_dw(x)
        x = self.concat_pw(x)
        return x


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Conv2DBN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ks=1,
                 stride=1,
                 pad=0,
                 dilation=1,
                 groups=1,
                 bn_weight_init=1,
                 lr_mult=1.0):
        super().__init__()
        conv_weight_attr = paddle.ParamAttr(learning_rate=lr_mult)
        self.c = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=ks,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=groups,
            weight_attr=conv_weight_attr,
            bias_attr=False)
        bn_weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.Constant(bn_weight_init),
            learning_rate=lr_mult)
        bn_bias_attr = paddle.ParamAttr(
            initializer=nn.initializer.Constant(0), learning_rate=lr_mult)
        self.bn = nn.BatchNorm2D(
            out_channels, weight_attr=bn_weight_attr, bias_attr=bn_bias_attr)

    def forward(self, inputs):
        out = self.c(inputs)
        out = self.bn(out)
        return out


class MLP(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.,
                 lr_mult=1.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2DBN(in_features, hidden_features, lr_mult=lr_mult)
        param_attr = paddle.ParamAttr(learning_rate=lr_mult)
        self.dwconv = nn.Conv2D(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            groups=hidden_features,
            weight_attr=param_attr,
            bias_attr=param_attr)
        self.act = act_layer()
        self.fc2 = Conv2DBN(hidden_features, out_features, lr_mult=lr_mult)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Layer):
    def __init__(self,
                 dim,
                 key_dim,
                 num_heads,
                 attn_ratio=4,
                 activation=None,
                 lr_mult=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2DBN(dim, nh_kd, 1, lr_mult=lr_mult)
        self.to_k = Conv2DBN(dim, nh_kd, 1, lr_mult=lr_mult)
        self.to_v = Conv2DBN(dim, self.dh, 1, lr_mult=lr_mult)

        self.proj = nn.Sequential(
            activation(),
            Conv2DBN(
                self.dh, dim, bn_weight_init=0, lr_mult=lr_mult))

    def forward(self, x):
        x_shape = paddle.shape(x)
        H, W = x_shape[2], x_shape[3]

        qq = self.to_q(x).reshape(
            [0, self.num_heads, self.key_dim, -1]).transpose([0, 1, 3, 2])
        kk = self.to_k(x).reshape([0, self.num_heads, self.key_dim, -1])
        vv = self.to_v(x).reshape([0, self.num_heads, self.d, -1]).transpose(
            [0, 1, 3, 2])

        attn = paddle.matmul(qq, kk)
        attn = F.softmax(attn, axis=-1)

        xx = paddle.matmul(attn, vv)

        xx = xx.transpose([0, 1, 3, 2]).reshape([0, self.dh, H, W])
        xx = self.proj(xx)
        return xx


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 key_dim,
                 num_heads,
                 mlp_ratios=4.,
                 attn_ratio=2.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.ReLU,
                 lr_mult=1.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratios = mlp_ratios

        # from paddleseg.models.rtformer import ExternalAttention 
        # self.attn = ExternalAttention(dim, dim, 256, num_heads=8, use_cross_kv=False) 
        self.attn = Attention(
           dim,
           key_dim=key_dim,
           num_heads=num_heads,
           attn_ratio=attn_ratio,
           activation=act_layer,
           lr_mult=lr_mult)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        mlp_hidden_dim = int(dim * mlp_ratios)
        self.mlp = MLP(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop,
                       lr_mult=lr_mult)

    def forward(self, x):
        h = x
        x = self.attn(x)
        x = self.drop_path(x)
        x = h + x

        h = x
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + h
        return x


class BasicLayer(nn.Layer):
    def __init__(self,
                 block_num,
                 embedding_dim,
                 key_dim,
                 num_heads,
                 mlp_ratios=4.,
                 attn_ratio=2.,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=None,
                 lr_mult=1.0):
        super().__init__()
        self.block_num = block_num

        self.transformer_blocks = nn.LayerList()
        for i in range(self.block_num):
            self.transformer_blocks.append(
                Block(
                    embedding_dim,
                    key_dim=key_dim,
                    num_heads=num_heads,
                    mlp_ratios=mlp_ratios,
                    attn_ratio=attn_ratio,
                    drop=drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list) else drop_path,
                    act_layer=act_layer,
                    lr_mult=lr_mult))

    def forward(self, x):
        # token * N 
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x


class PyramidPoolAgg(nn.Layer):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride
        self.tmp = Identity()  # avoid the error of paddle.flops

    def forward(self, inputs):
        '''
        # The F.adaptive_avg_pool2d does not support the (H, W) be Tensor,
        # so exporting the inference model will raise error.
        _, _, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        return paddle.concat(
            [F.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], axis=1)
        '''
        out = []
        ks = 2**(len(inputs)-1)
        stride = self.stride**(len(inputs)-1)
        for x in inputs[:-1]:
            x = F.avg_pool2d(x, int(ks), int(stride))
            ks /= 2
            stride /= 2
            out.append(x)
        else:
            out.append(inputs[-1])
        out = paddle.concat(out, axis=1)
        return out

class ESNet(TheseusLayer):
    def __init__(self,
                 stages_pattern,
                 class_num=1000,
                 scale=1.0,
                 dropout_prob=0.2,
                 class_expand=1280,
                 return_patterns=None,
                 return_stages=None):
        super().__init__()
        self.scale = scale
        self.class_num = class_num
        self.class_expand = class_expand
        stage_repeats = [3, 7, 3]
        stage_out_channels = [
            -1, 24, make_divisible(116 * scale), make_divisible(232 * scale), # channels: [24,120,232,464] topformer channels [32,64,128,160]
            make_divisible(464 * scale), 1024
        ]

        self.conv1 = ConvBNLayer(
            in_channels=3,
            out_channels=stage_out_channels[1],
            kernel_size=3,
            stride=2)
        self.max_pool = MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.block_list = nn.LayerList()
        for stage_id, num_repeat in enumerate(stage_repeats): # 2, 9, 12
            for i in range(num_repeat):
                if i == 0:
                    block = ESBlock2(
                        in_channels=stage_out_channels[stage_id + 1],
                        out_channels=stage_out_channels[stage_id + 2])
                else:
                    block = ESBlock1(
                        in_channels=stage_out_channels[stage_id + 2],
                        out_channels=stage_out_channels[stage_id + 2])
                self.block_list.append(block)
        # self.blocks = nn.Sequential(*block_list)

        injection_out_channels=[None, 256, 256, 256]
        encoder_out_indices=[2, 4, 6, 9]
        trans_out_indices=[1, 2, 3]
        depths=4
        key_dim=16
        num_heads=8
        attn_ratios=2
        mlp_ratios=2
        c2t_stride=2
        drop_path_rate=0.
        act_layer=nn.ReLU6
        injection_type="multi_sum"
        injection=True
        lr_mult=0.1

        self.embed_dim = 640 # 384 -> 432

        self.ppa = PyramidPoolAgg(stride=c2t_stride)

        dpr = [x.item() for x in \
               paddle.linspace(0, drop_path_rate, depths)]
        self.trans = BasicLayer(
            block_num=depths,
            embedding_dim=self.embed_dim,
            key_dim=key_dim,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            attn_ratio=attn_ratios,
            drop=0,
            attn_drop=0,
            drop_path=dpr,
            act_layer=act_layer,
            lr_mult=lr_mult)

        self.conv2 = ConvBNLayer(
            in_channels=self.embed_dim,# stage_out_channels[-2],
            out_channels=stage_out_channels[-1],
            kernel_size=1)

        self.avg_pool = AdaptiveAvgPool2D(1)

        self.last_conv = Conv2D(
            in_channels=stage_out_channels[-1],
            out_channels=self.class_expand,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=False)
        self.hardswish = nn.Hardswish()
        self.dropout = Dropout(p=dropout_prob, mode="downscale_in_infer")
        self.flatten = nn.Flatten(start_axis=1, stop_axis=-1)
        self.fc = Linear(self.class_expand, self.class_num)

        super().init_res(
            stages_pattern,
            return_patterns=return_patterns,
            return_stages=return_stages)

    def forward(self, x):
        out_indices=[2, 9, 12]
        x = self.conv1(x)
        x = self.max_pool(x) # x4
        outputs = [x]
        for i, block in enumerate(self.block_list):
            x = block(x)
            if i in out_indices:
                outputs.append(x)
        # x = self.blocks(x)   # channels: [24,56,120,232] topformer channels [32,64,128,160] # 直接4倍下采样的，细节信息可能不是很好
        # 0.5 [512, 24, 56, 56] [512, 56, 28, 28] [512, 120, 14, 14] [512, 232, 7, 7]
        out = self.ppa(outputs) #1.0: [512, 24, 56, 56];[512, 120, 28, 28][512, 232, 14, 14][512, 464, 7, 7] 
        out = self.trans(out) # [512, 432, 3, 3]
        x = self.conv2(out)
        x = self.avg_pool(x)
        x = self.last_conv(x)
        x = self.hardswish(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def _load_pretrained(pretrained, model, model_url, use_ssld):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def ESNet_x0_25(pretrained=False, use_ssld=False, **kwargs):
    """
    ESNet_x0_25
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `ESNet_x0_25` model depends on args.
    """
    model = ESNet(
        scale=0.25, stages_pattern=MODEL_STAGES_PATTERN["ESNet"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["ESNet_x0_25"], use_ssld)
    return model


def ESNet_x0_5(pretrained=False, use_ssld=False, **kwargs):
    """
    ESNet_x0_5
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `ESNet_x0_5` model depends on args.
    """
    model = ESNet(
        scale=0.5, stages_pattern=MODEL_STAGES_PATTERN["ESNet"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["ESNet_x0_5"], use_ssld)
    return model


def ESNet_x0_75(pretrained=False, use_ssld=False, **kwargs):
    """
    ESNet_x0_75
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `ESNet_x0_75` model depends on args.
    """
    model = ESNet(
        scale=0.75, stages_pattern=MODEL_STAGES_PATTERN["ESNet"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["ESNet_x0_75"], use_ssld)
    return model


def ESNet_x1_0(pretrained=False, use_ssld=False, **kwargs):
    """
    ESNet_x1_0
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `ESNet_x1_0` model depends on args.
    """
    model = ESNet(
        scale=1.0, stages_pattern=MODEL_STAGES_PATTERN["ESNet"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["ESNet_x1_0"], use_ssld)
    return model
