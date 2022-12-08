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

# reference: https://arxiv.org/abs/1905.02244

from __future__ import absolute_import, division, print_function

import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.nn import AdaptiveAvgPool2D, BatchNorm, Conv2D, Dropout, Linear
from paddle.regularizer import L2Decay
import paddle.nn.functional as F

from ..base.theseus_layer import TheseusLayer
from ....utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {
    "MobileNetV3_small_x0_35":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x0_35_pretrained.pdparams",
    "MobileNetV3_small_x0_5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x0_5_pretrained.pdparams",
    "MobileNetV3_small_x0_75":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x0_75_pretrained.pdparams",
    "MobileNetV3_small_x1_0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x1_0_pretrained.pdparams",
    "MobileNetV3_small_x1_25":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x1_25_pretrained.pdparams",
    "MobileNetV3_large_x0_35":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x0_35_pretrained.pdparams",
    "MobileNetV3_large_x0_5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x0_5_pretrained.pdparams",
    "MobileNetV3_large_x0_75":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x0_75_pretrained.pdparams",
    "MobileNetV3_large_x1_0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x1_0_pretrained.pdparams",
    "MobileNetV3_large_x1_25":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x1_25_pretrained.pdparams",
}

MODEL_STAGES_PATTERN = {
    "MobileNetV3_small":
    ["blocks[0]", "blocks[2]", "blocks[7]", "blocks[10]"],
    "MobileNetV3_large":
    ["blocks[0]", "blocks[2]", "blocks[5]", "blocks[11]", "blocks[14]"],
    "MobileNetV3_large_edit":
    ["blocks[0]", "blocks[2]", "blocks[4]", "blocks[6]", "blocks[9]"]
}

__all__ = MODEL_URLS.keys()

# "large", "small" is just for MobinetV3_large, MobileNetV3_small respectively.
# The type of "large" or "small" config is a list. Each element(list) represents a depthwise block, which is composed of k, exp, se, act, s.
# k: kernel_size
# exp: middle channel number in depthwise block
# c: output channel number in depthwise block
# se: whether to use SE block
# act: which activation to use
# s: stride in depthwise block
NET_CONFIG = {
    "large": [
        # k, exp, c, se, act, s
        [3, 16, 16, False, "relu", 1],
        [3, 64, 24, False, "relu", 2],
        [3, 72, 24, False, "relu", 1],# 2
        [5, 72, 40, True, "relu", 2],
        [5, 120, 40, True, "relu", 1],
        [5, 120, 40, True, "relu", 1],# 5
        [3, 240, 80, False, "hardswish", 2],
        [3, 200, 80, False, "hardswish", 1],
        [3, 184, 80, False, "hardswish", 1],
        [3, 184, 80, False, "hardswish", 1],
        [3, 480, 112, True, "hardswish", 1],
        [3, 672, 112, True, "hardswish", 1],# 11
        [5, 672, 160, True, "hardswish", 2],
        [5, 960, 160, True, "hardswish", 1],
        [5, 960, 160, True, "hardswish", 1],# 14
    ],
    "large_edit": [
        # k, exp, c, se, act, s
        [3, 16, 16, False, "relu", 1],
        [3, 64, 32, False, "relu", 2],
        [3, 72, 32, False, "relu", 1],# 2
        [5, 72, 64, True, "relu", 2],
        [5, 128, 64, True, "relu", 1], # 4
        [3, 384, 128, False, "hardswish", 2],
        [3, 480, 128, False, "hardswish", 1],# 6
        [5, 480, 160, True, "hardswish", 2],
        [5, 960, 160, True, "hardswish", 1],
        [5, 960, 160, True, "hardswish", 1],# 9
    ],
    "small": [
        # k, exp, c, se, act, s
        [3, 16, 16, True, "relu", 2],
        [3, 72, 24, False, "relu", 2],
        [3, 88, 24, False, "relu", 1],
        [5, 96, 40, True, "hardswish", 2],
        [5, 240, 40, True, "hardswish", 1],
        [5, 240, 40, True, "hardswish", 1],
        [5, 120, 48, True, "hardswish", 1],
        [5, 144, 48, True, "hardswish", 1],
        [5, 288, 96, True, "hardswish", 2],
        [5, 576, 96, True, "hardswish", 1],
        [5, 576, 96, True, "hardswish", 1],
    ]
}
# first conv output channel number in MobileNetV3
STEM_CONV_NUMBER = 16
# last second conv output channel for "small"
LAST_SECOND_CONV_SMALL = 576
# last second conv output channel for "large"
LAST_SECOND_CONV_LARGE = 960
# last conv output channel number for "large" and "small"
LAST_CONV = 1280


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _create_act(act):
    if act == "hardswish":
        return nn.Hardswish()
    elif act == "relu":
        return nn.ReLU()
    elif act is None:
        return None
    else:
        raise RuntimeError(
            "The activation function is not supported: {}".format(act))



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
        ks = 2**len(inputs)
        stride = self.stride**len(inputs)
        for x in inputs:
            x = F.avg_pool2d(x, int(ks), int(stride))
            ks /= 2
            stride /= 2
            out.append(x)
        out = paddle.concat(out, axis=1)
        return out


class MobileNetV3(TheseusLayer):
    """
    MobileNetV3
    Args:
        config: list. MobileNetV3 depthwise blocks config.
        scale: float=1.0. The coefficient that controls the size of network parameters. 
        class_num: int=1000. The number of classes.
        inplanes: int=16. The output channel number of first convolution layer.
        class_squeeze: int=960. The output channel number of penultimate convolution layer. 
        class_expand: int=1280. The output channel number of last convolution layer. 
        dropout_prob: float=0.2.  Probability of setting units to zero.
    Returns:
        model: nn.Layer. Specific MobileNetV3 model depends on args.
    """

    def __init__(self,
                 config,
                 stages_pattern,
                 scale=1.0,
                 class_num=1000,
                 inplanes=STEM_CONV_NUMBER,
                 class_squeeze=LAST_SECOND_CONV_LARGE,
                 class_expand=LAST_CONV,
                 dropout_prob=0.2,
                 out_index=None,
                 return_patterns=None,
                 return_stages=None,
                 **kwargs):
        super().__init__()

        self.cfg = config
        self.scale = scale
        self.inplanes = inplanes
        self.class_squeeze = class_squeeze
        self.class_expand = class_expand
        self.class_num = class_num

        self.conv = ConvBNLayer(
            in_c=3,
            out_c=_make_divisible(self.inplanes * self.scale),
            filter_size=3,
            stride=2,
            padding=1,
            num_groups=1,
            if_act=True,
            act="hardswish")

        self.blocks = nn.LayerList()
        for i, (k, exp, c, se, act, s) in enumerate(self.cfg):
            self.blocks.append(ResidualUnit(
                in_c=_make_divisible(self.inplanes * self.scale if i == 0 else
                                     self.cfg[i - 1][2] * self.scale),
                mid_c=_make_divisible(self.scale * exp),
                out_c=_make_divisible(self.scale * c),
                filter_size=k,
                stride=s,
                use_se=se,
                act=act))

        # self.blocks = nn.Sequential(* [
        #     ResidualUnit(
        #         in_c=_make_divisible(self.inplanes * self.scale if i == 0 else
        #                              self.cfg[i - 1][2] * self.scale),
        #         mid_c=_make_divisible(self.scale * exp),
        #         out_c=_make_divisible(self.scale * c),
        #         filter_size=k,
        #         stride=s,
        #         use_se=se,
        #         act=act) for i, (k, exp, c, se, act, s) in enumerate(self.cfg)
        # ])
        injection_out_channels=[None, 256, 256, 256]
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
        self.out_index = out_index

        self.embed_dim = 336 # 384-> 432  # TODO 384

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

        self.last_second_conv = ConvBNLayer(
            in_c=_make_divisible(self.embed_dim * self.scale), #_make_divisible(self.cfg[-1][2] * self.scale),
            out_c=_make_divisible(self.scale * self.class_squeeze),
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            if_act=True,
            act="hardswish")

        self.avg_pool = AdaptiveAvgPool2D(1)

        self.last_conv = Conv2D(
            in_channels=_make_divisible(self.scale * self.class_squeeze),
            out_channels=self.class_expand,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=False)

        self.hardswish = nn.Hardswish()
        if dropout_prob is not None:
            self.dropout = Dropout(p=dropout_prob, mode="downscale_in_infer")
        else:
            self.dropout = None
        self.flatten = nn.Flatten(start_axis=1, stop_axis=-1)

        self.fc = Linear(self.class_expand, class_num)

        super().init_res(
            stages_pattern,
            return_patterns=return_patterns,
            return_stages=return_stages)

    def forward(self, x):
        x = self.conv(x)
        outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.out_index:
                outputs.append(x)
        # x = self.blocks(x)
        out = self.ppa(outputs) #1.0: [512, 24, 56, 56];[512, 120, 28, 28][512, 232, 14, 14][512, 464, 7, 7] 
        out = self.trans(out) # [512, 432, 3, 3]
        x = self.last_second_conv(out)
        x = self.avg_pool(x)
        x = self.last_conv(x)
        x = self.hardswish(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


class ConvBNLayer(TheseusLayer):
    def __init__(self,
                 in_c,
                 out_c,
                 filter_size,
                 stride,
                 padding,
                 num_groups=1,
                 if_act=True,
                 act=None):
        super().__init__()

        self.conv = Conv2D(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            bias_attr=False)
        self.bn = BatchNorm(
            num_channels=out_c,
            act=None,
            param_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.if_act = if_act
        self.act = _create_act(act)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            x = self.act(x)
        return x


class ResidualUnit(TheseusLayer):
    def __init__(self,
                 in_c,
                 mid_c,
                 out_c,
                 filter_size,
                 stride,
                 use_se,
                 act=None):
        super().__init__()
        self.if_shortcut = stride == 1 and in_c == out_c
        self.if_se = use_se

        self.expand_conv = ConvBNLayer(
            in_c=in_c,
            out_c=mid_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act=act)
        self.bottleneck_conv = ConvBNLayer(
            in_c=mid_c,
            out_c=mid_c,
            filter_size=filter_size,
            stride=stride,
            padding=int((filter_size - 1) // 2),
            num_groups=mid_c,
            if_act=True,
            act=act)
        if self.if_se:
            self.mid_se = SEModule(mid_c)
        self.linear_conv = ConvBNLayer(
            in_c=mid_c,
            out_c=out_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None)

    def forward(self, x):
        identity = x
        x = self.expand_conv(x)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = paddle.add(identity, x)
        return x


# nn.Hardsigmoid can't transfer "slope" and "offset" in nn.functional.hardsigmoid
class Hardsigmoid(TheseusLayer):
    def __init__(self, slope=0.2, offset=0.5):
        super().__init__()
        self.slope = slope
        self.offset = offset

    def forward(self, x):
        return nn.functional.hardsigmoid(
            x, slope=self.slope, offset=self.offset)


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
        self.hardsigmoid = Hardsigmoid(slope=0.2, offset=0.5)

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        return paddle.multiply(x=identity, y=x)


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


def MobileNetV3_small_x0_35(pretrained=False, use_ssld=False, **kwargs):
    """
    MobileNetV3_small_x0_35
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV3_small_x0_35` model depends on args.
    """
    model = MobileNetV3(
        config=NET_CONFIG["small"],
        scale=0.35,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_small"],
        class_squeeze=LAST_SECOND_CONV_SMALL,
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV3_small_x0_35"],
                     use_ssld)
    return model


def MobileNetV3_small_x0_5(pretrained=False, use_ssld=False, **kwargs):
    """
    MobileNetV3_small_x0_5
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV3_small_x0_5` model depends on args.
    """
    model = MobileNetV3(
        config=NET_CONFIG["small"],
        scale=0.5,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_small"],
        class_squeeze=LAST_SECOND_CONV_SMALL,
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV3_small_x0_5"],
                     use_ssld)
    return model


def MobileNetV3_small_x0_75(pretrained=False, use_ssld=False, **kwargs):
    """
    MobileNetV3_small_x0_75
    Args:
        pretrained: bool=false or str. if `true` load pretrained parameters, `false` otherwise.
                    if str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV3_small_x0_75` model depends on args.
    """
    model = MobileNetV3(
        config=NET_CONFIG["small"],
        scale=0.75,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_small"],
        class_squeeze=LAST_SECOND_CONV_SMALL,
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV3_small_x0_75"],
                     use_ssld)
    return model


def MobileNetV3_small_x1_0(pretrained=False, use_ssld=False, **kwargs):
    """
    MobileNetV3_small_x1_0
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV3_small_x1_0` model depends on args.
    """
    model = MobileNetV3(
        config=NET_CONFIG["small"],
        scale=1.0,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_small"],
        class_squeeze=LAST_SECOND_CONV_SMALL,
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV3_small_x1_0"],
                     use_ssld)
    return model


def MobileNetV3_small_x1_25(pretrained=False, use_ssld=False, **kwargs):
    """
    MobileNetV3_small_x1_25
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV3_small_x1_25` model depends on args.
    """
    model = MobileNetV3(
        config=NET_CONFIG["small"],
        scale=1.25,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_small"],
        class_squeeze=LAST_SECOND_CONV_SMALL,
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV3_small_x1_25"],
                     use_ssld)
    return model


def MobileNetV3_large_x0_35(pretrained=False, use_ssld=False, **kwargs):
    """
    MobileNetV3_large_x0_35
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV3_large_x0_35` model depends on args.
    """
    model = MobileNetV3(
        config=NET_CONFIG["large"],
        scale=0.35,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_small"],
        class_squeeze=LAST_SECOND_CONV_LARGE,
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV3_large_x0_35"],
                     use_ssld)
    return model


def MobileNetV3_large_x0_5(pretrained=False, use_ssld=False, **kwargs):
    """
    MobileNetV3_large_x0_5
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV3_large_x0_5` model depends on args.
    """
    model = MobileNetV3(
        config=NET_CONFIG["large"],
        scale=0.5,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_large"],
        class_squeeze=LAST_SECOND_CONV_LARGE,
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV3_large_x0_5"],
                     use_ssld)
    return model


def MobileNetV3_large_x0_75(pretrained=False, use_ssld=False, **kwargs):
    """
    MobileNetV3_large_x0_75
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV3_large_x0_75` model depends on args.
    """
    model = MobileNetV3(
        config=NET_CONFIG["large"],
        scale=0.75,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_large"],
        class_squeeze=LAST_SECOND_CONV_LARGE,
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV3_large_x0_75"],
                     use_ssld)
    return model


def MobileNetV3_large_x1_0(pretrained=False, use_ssld=False, **kwargs):
    """
    MobileNetV3_large_x1_0
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV3_large_x1_0` model depends on args.
    """
    model = MobileNetV3(
        config=NET_CONFIG["large"],
        scale=1.0,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_large"],
        class_squeeze=LAST_SECOND_CONV_LARGE,
        out_index=[2,5,11,14],
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV3_large_x1_0"],
                     use_ssld)
    return model

def MobileNetV3_large_x1_0_edit(pretrained=False, use_ssld=False, **kwargs):
    """
    MobileNetV3_large_x1_0
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV3_large_x1_0` model depends on args.
    """
    model = MobileNetV3(
        config=NET_CONFIG["large_edit"],
        scale=1.0,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_large"],
        class_squeeze=LAST_SECOND_CONV_LARGE,
        out_index=[2,4,6,9],
        **kwargs)
    # _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV3_large_x1_0"],
    #                  use_ssld)
    return model


def MobileNetV3_large_x1_25(pretrained=False, use_ssld=False, **kwargs):
    """
    MobileNetV3_large_x1_25
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV3_large_x1_25` model depends on args.
    """
    model = MobileNetV3(
        config=NET_CONFIG["large"],
        scale=1.25,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_large"],
        class_squeeze=LAST_SECOND_CONV_LARGE,
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV3_large_x1_25"],
                     use_ssld)
    return model
