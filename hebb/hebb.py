import torch
import torch.nn as nn

from .functional import *
import params as P


# TODO:
#   - 基于层归一化而不是批归一化的归一化，修改批归一化层，使其仅适应性地计算统计量，而不是按批次计算，
#       并且在反向传播时计算梯度，仿佛统计量是按批次计算的。同样适用于其他自适应参数。可以选择用方差的平均值
#       来归一化，而不是每个特征用其自己的方差。可以选择基于权重的启发式方法进行归一化。
#   - 添加延迟更新标志和模块内的局部优化。合并所有自适应参数的更新。
#   - 非线性后解混合ICA，混合ICA和PCA，带有偏置的重建。
#   - 高斯非线性聚类，集成向量投影相似性，使用权重向量作为均值编码和偏置作为方差，这样可以找到基于点积相似性
#       的对齐簇。
#   - 考虑去除可能由填充引起的边界伪影。
#   - 其他赫布方法。

# 一个通用的归一化层
class GenNorm(nn.Module):

    def __init__(self, n, beta=0.1, eps=1e-9, affine=True):
        super(GenNorm, self).__init__()

        # 定义可训练的偏置参数
        self.bias = nn.Parameter(torch.zeros(n).float(), requires_grad=True)
        # 注册一个用于跟踪运行均值的缓冲区
        self.register_buffer('running_mean', torch.zeros(n).float())
        # self.running_mean = nn.Parameter(torch.zeros(n).float(), requires_grad=True)
        # 定义可训练的权重参数
        self.weight = nn.Parameter(torch.ones(n).float(), requires_grad=True)
        # 注册一个用于跟踪运行方差的缓冲区
        self.register_buffer('running_var', torch.ones(n).float())

        self.beta = beta  # 动量参数
        self.eps = eps  # 用于数值稳定的小常数
        self.affine = affine  # 是否使用可学习的仿射变换参数

    def track(self, x):
        """
        更新运行均值和运行方差。
        """
        mean = x.mean(dim=(0, 2))  # 计算当前批次的均值
        var = x.var(dim=(0, 2))  # 计算当前批次的方差
        # 更新运行均值和运行方差
        self.running_mean = self.running_mean + self.beta * (mean - self.running_mean)
        self.running_var = self.running_var + self.beta * (var - self.running_var)

    def normalize(self, x):
        """
        归一化输入x。
        """
        res = (x - self.running_mean.view(1, -1, 1)) / torch.sqrt(self.running_var.view(1, -1, 1) + self.eps)
        if self.affine:
            res = res * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return res

    def forward(self, x):
        orig_size = x.size()
        # 将输入x重新整形为 (batch大小, 通道数, 输入/窗口大小)
        x = x.view(x.size(0), x.size(1), -1)
        # 在不计算梯度的情况下跟踪均值和方差
        with torch.no_grad():
            self.track(x)
        # 归一化并恢复输入的原始形状
        return self.normalize(x).view(orig_size)

# def parameters(self, recurse: bool = ...):
#    return [self.weight, self.bias]


# 一个竞争激活层
class Competitive(nn.Module):
    # 随机放弃策略的类型
    HARD_RAND_ABST = 'hard_rand_abst'  # 硬随机放弃
    SOFT_RAND_ABST = 'soft_rand_abst'  # 软随机放弃

    # LFB（侧向反馈）核的类型
    LFB_GAUSS = 'lfb_gauss'  # 高斯
    LFB_DoG = 'lfb_DoG'  # 差分高斯
    LFB_EXP = 'lfb_exp'  # 指数
    LFB_DoE = 'lfb_DoE'  # 指数差分

    # k参数的适应机制类型
    ADA_K_MODE_STD = 'ada_k_mode_std'  # 标准
    ADA_K_MODE_LOG = 'ada_k_mode_log'  # 对数
    ADA_K_MODE_SHIFT = 'ada_k_mode_shift'  # 平移

    def __init__(self,
                 out_size=None,
                 competitive_act=None,
                 k=1,
                 lrn_k=False,
                 ada_k=None,
                 random_abstention=None,
                 y_gating=False,
                 lfb_y_gating=False,
                 lfb_value=None,
                 lfb_sigma=None,
                 lfb_tau=1000,
                 beta=.1):
        # 默认情况下输出全为1，即所有都是赢家（平凡竞争）。Competitive(lfb_y_gating=True)给出恒等映射。
        super(Competitive, self).__init__()
        # 启用/禁用随机放弃、竞争学习、侧向反馈等特性
        self.competitive_act = competitive_act
        self.competitive = self.competitive_act is not None
        self.beta = beta
        self.k = k if not lrn_k else nn.Parameter(torch.tensor(float(k)), requires_grad=True)
        if ada_k not in [None, self.ADA_K_MODE_STD, self.ADA_K_MODE_LOG, self.ADA_K_MODE_SHIFT]:
            raise ValueError("Invalid value for argument ada_k: " + str(ada_k))
        self.ada_k = ada_k
        self.trk = nn.BatchNorm2d(1, momentum=self.beta, affine=False)
        if random_abstention not in [None, self.SOFT_RAND_ABST, self.HARD_RAND_ABST]:
            raise ValueError("Invalid value for argument random_abstention: " + str(random_abstention))
        self.random_abstention = random_abstention
        self.random_abstention_on = self.competitive and self.random_abstention is not None
        self.y_gating = y_gating
        self.lfb_y_gating = lfb_y_gating
        self.lfb_value = lfb_value
        self.lfb_on = self.lfb_value is not None and self.lfb_value != 0

        # 初始化输出大小，只有在启用随机放弃或侧向反馈时才需要
        self.out_size = None
        self.out_channels = None
        if self.random_abstention_on or self.lfb_on:
            if out_size is None:
                raise ValueError("Invalid value for argument out_size: " + str(
                    out_size) + " when random abstention or lfb is provided")
            if hasattr(out_size, '__len__') and len(out_size) > 3:
                raise ValueError("Too many dimensions for argument out_size: " + str(out_size) + " (up to 3 allowed)")
            out_size_list = [out_size] if not hasattr(out_size, '__len__') else out_size
            self.out_size = torch.tensor(out_size_list)
            self.out_channels = self.out_size.prod().item()

        # 设置与侧向反馈相关的参数
        if self.lfb_on:
            # 准备生成将用于应用侧向反馈的核的变量
            map_radius = (self.out_size - 1) // 2
            lfb_sigma = map_radius.max().item() if lfb_sigma is None else lfb_sigma
            x = torch.abs(torch.arange(0, self.out_size[0].item()) - map_radius[0])
            for i in range(1, self.out_size.size(0)):
                x_new = torch.abs(torch.arange(0, self.out_size[i].item()) - map_radius[i])
                for j in range(i): x_new = x_new.unsqueeze(j)
                x = torch.max(x.unsqueeze(-1), x_new)  # max给出L_infinity距离，sum给出L_1距离，root_p(sum x^p)给出L_p距离
            # 存储将用于应用侧向反馈的核在注册缓冲区中
            if lfb_value == self.LFB_EXP or lfb_value == self.LFB_DoE:
                self.register_buffer('lfb_kernel', torch.exp(-x.float() / lfb_sigma))
            if lfb_value == self.LFB_GAUSS or lfb_value == self.LFB_DoG:
                self.register_buffer('lfb_kernel', torch.exp(-x.pow(2).float() / (2 * (lfb_sigma ** 2))))
            else:  # lfb_value是一个数字
                if type(lfb_value) is not int and type(lfb_value) is not float:
                    raise ValueError("Invalid value for argument lfb_value: " + str(lfb_value))
                self.register_buffer('lfb_kernel', (x == 0).float())
                x[x == 0] = lfb_value
            # 在应用lfb核之前填充输入的填充
            pad_pre = map_radius.unsqueeze(1)
            pad_post = (self.out_size - 1 - map_radius).unsqueeze(1)
            self.pad = list(torch.cat((pad_pre, pad_post), dim=1).flip(0).view(-1))
            # LFB核收缩参数
            self.gamma = torch.exp(
                torch.log(torch.tensor(lfb_sigma).float()) / lfb_tau).item() if lfb_tau is not None else None
            if (
                    lfb_value == self.LFB_GAUSS or lfb_value == self.LFB_DoG) and self.gamma is not None: self.gamma = self.gamma ** 2
        else:
            self.register_buffer('lfb_kernel', None)

        # 初始化统计收集的变量
        if self.random_abstention_on:
            self.register_buffer('victories_count', torch.zeros(self.out_channels).float())
        else:
            self.register_buffer('victories_count', None)

    def get_k(self, scores, lrn=False):
        """
        获取k值，如果处于训练模式并启用了学习，则跟踪统计数据。
        """
        k = self.k
        if self.training and lrn:  # 跟踪统计数据
            if self.ada_k == self.ADA_K_MODE_LOG:
                _ = self.trk(torch.log(scores).view(-1, 1, 1, 1))
            else:
                if self.ada_k is not None:  # 注意：当我们实现更高效的跟踪时，这个条件将被移除（目前速度较慢）
                    _ = self.trk(scores.view(-1, 1, 1, 1))

        # 计算自适应的k值
        if self.ada_k in [self.ADA_K_MODE_STD, self.ADA_K_MODE_LOG]:
            k = k * self.trk.running_var ** 0.5
        if self.ada_k in [self.ADA_K_MODE_SHIFT]:
            k = -self.trk.running_mean + k * self.trk.running_var ** 0.5
        return k

    def forward(self, y, t=None, lrn=False):
        """
        前向传播函数
        """
        # 随机放弃
        scores = y
        if self.random_abstention_on:
            abst_prob = self.victories_count / (self.victories_count.max() + y.size(0) / y.size(1)).clamp(1)
            if self.random_abstention == self.SOFT_RAND_ABST: scores = y * abst_prob.unsqueeze(0)
            if self.random_abstention == self.HARD_RAND_ABST:
                scores = y * (torch.rand_like(abst_prob) >= abst_prob).float().unsqueeze(0)

        # 竞争，返回的winner_mask是一个位图，表示神经元获胜和失败的位置。
        if self.competitive:
            winner_mask = self.competitive_act(scores, self.get_k(scores, lrn=lrn), t)
            if lrn and self.random_abstention_on and self.training:  # 使用随机放弃时更新统计数据
                winner_mask_sum = winner_mask.sum(0)  # 神经元获胜的输入数量
                self.victories_count += winner_mask_sum
                self.victories_count -= self.victories_count.min().item()
        else:
            winner_mask = torch.ones_like(y)

        # 如果需要，通过输出应用winner_mask门控
        if self.y_gating:
            winner_mask = winner_mask * y

        # 侧向反馈
        if self.lfb_on:
            lfb_kernel = self.lfb_kernel
            if self.lfb_value == self.LFB_DoG or self.lfb_value == self.LFB_DoE:
                lfb_kernel = 2 * lfb_kernel - lfb_kernel.pow(0.5)  # 高斯/指数差分（墨西哥帽形函数）

            lfb_in = F.pad(winner_mask.view(-1, *self.out_size), self.pad)
            if self.out_size.size(0) == 1:
                lfb_out = torch.conv1d(lfb_in.unsqueeze(1), lfb_kernel.unsqueeze(0).unsqueeze(1))
            elif self.out_size.size(0) == 2:
                lfb_out = torch.conv2d(lfb_in.unsqueeze(1), lfb_kernel.unsqueeze(0).unsqueeze(1))
            else:
                lfb_out = torch.conv3d(lfb_in.unsqueeze(1), lfb_kernel.unsqueeze(0).unsqueeze(1))
            lfb_out = lfb_out.clamp(-1, 1).view_as(y)
        else:
            lfb_out = winner_mask

        # 如果需要，通过输出应用lfb门控
        if self.lfb_y_gating:
            lfb_out = lfb_out * y

        # LFB核收缩调度
        if lrn and self.lfb_on and self.gamma is not None and self.training:
            self.lfb_kernel = self.lfb_kernel.pow(self.gamma)

        return lfb_out


# 这个模块表示一层使用赫布算法训练的卷积神经元
class HebbianConv2d(nn.Module):
    # s = sim(w, x), y = act(s) -- 例如：s = w^T x, y = f(s)

    # 门控项的类型
    GATE_BASE = 'gate_base'  # r = cmp_res
    GATE_HEBB = 'gate_hebb'  # r = cmp_res * y
    GATE_DIFF = 'gate_diff'  # r = cmp_res - y
    GATE_SMAX = 'gate_smx'  # r = cmp_res - softmax(y)

    # 重建方案的类型
    REC_QNT = 'rec_qnt'  # reconstr = w
    REC_QNT_SGN = 'rec_qnt_sgn'  # reconstr = sign(cmp_res) * w
    REC_LIN_CMB = 'rec_lin_cmb'  # reconstr = sum_i r_i w_i

    # 更新步骤的类型
    UPD_RECONSTR = 'upd_reconstr'  # delta_w = alpha * r * (x - reconstr)
    UPD_ICA = 'upd_ica'  # delta w_i = r_i (w_i - y_i rs^T W) # 注意：r项作为门控
    UPD_HICA = 'upd_hica'  # delta w_i = r_i (w_i - y_i sum_(k=1..i) rs_k w_k)
    UPD_ICA_NRM = 'upd_ica_nrm'  # delta w_i = r_i (w_i w_i^T - I) y_i rs^T W
    UPD_HICA_NRM = 'upd_hica_nrm'  # delta w_i = r_i (w_i w_i^T - I) y_i sum_(k=1..i) rs_k w_k

    # ICA规则
    # Delta W = (I - f(s) s^T) W
    # Delta W_i = sum_k (I - f(s) s^T)_ik W_k = sum_k (delta_ik - f(s_i) s_k) W_k
    # = delta_ii W_i - f(s_i) sum_k s_k W_k = W_i - f(s_i) sum_k s_k W_k  = W_i - f(s_i) s^T W
    # 带归一化的ICA：动态估计sigma
    # f(s) = (log(p(s/sigma)))' = (1/sigma) * ( p'(s/sigma)/p(s/sigma) ) = (1/sigma) * phi(s/sigma)
    # 其中 phi(s) = p'(s)/p(s)
    # 重写：Delta W = (I - f(s) s^T) W = (I - (1/sigma^2) * sigma^2 f(s) s^T) W  -
    # [注意 Delta W ~ (sigma^2 - g(s) s^T) W -- 其中 g(s) = sigma^s f(s)]
    # Delta W = (I - f(s) s^T) W
    # Delta W_i = sum_k (I - f(s) s^T)_ik W_k = sum_k (delta_ik - f(s_i) s_k) W_k
    # = W_i - f(s_i) sum_k s_k W_k
    # 归一化
    # w_new = (w + eta Delta w) / |w + eta Delta w|
    # |w + eta Delta w|^2 = w^2 (1 + 2 eta w^T Delta w) + o(eta^2)
    # (|w + eta Delta w|^2)^-1/2 = |w| (1 - eta w^T Delta w) + o(eta^2)
    # w_new = (w + eta Delta w)(w^-2 (1 - eta w^T Delta w) + o(eta^2))
    # = (w + eta (Delta w - (w^t Delta w) w) w^-2 + o(eta^2)
    # 假设w归一化 --> w^-2 = 1
    # w_new = w + eta (Delta w - (w^T Delta w) w) + o(eta^2)
    # 修正的 Delta w = Delta w - (w^T Delta w) w --> x y - y^2 w for oja
    # 在自适应方差ICA的情况下
    # 修正的 Delta w_i = w_i - f(s_i) sum_k s_k w_k - w_i(w_i^2 - f(s_i) sum_k s_k (w_k w_i^T)) ** 注意：w_i 是行向量
    # = w_i - w_i -f(s_i) sum_k s_k (w_k - w_i (w_k w_i^T)) = -f(s_i) sum_(k!=i) s_k (w_k - w_i (w_k w_i^T))
    # = -f(s_i) sum_k s_k (w_k - w_k w_i^T w_i) = -f(s_i) sum_k s_k w_k (I - w_i^T w_i) = -f(s_i) s^T W (I - w_i w_i^T)
    # = f(s_i) s^T W w_i^T w_i - f(s_i) s^T W = f(s_i) s^T W (w_i^T w_i - I)

    # 更新缩减方案的类型
    RED_AVG = 'red_avg'  # 平均
    RED_W_AVG = 'red_w_avg'  # 加权平均

    # 偏置更新方案的类型
    BIAS_MODE_BASE = 'bias_mode_base'
    BIAS_MODE_TARG = 'bias_mode_targ'
    BIAS_MODE_STD = 'bias_mode_std'
    BIAS_MODE_PERC = 'bias_mode_perc'
    BIAS_MODE_EXP = 'bias_mode_exp'
    BIAS_MODE_VALUE = 'bias_mode_value'

    # 激活互补模式 - 激活互补将一些神经元的非线性变换为x - 非线性。
    # 这可以在ICA训练期间用于超高斯和次高斯变量的不同非线性。
    ACT_COMPLEMENT_INIT_RAND = 'act_complement_init_rand'
    ACT_COMPLEMENT_INIT_SPLT = 'act_complement_init_splt'
    ACT_COMPLEMENT_INIT_ALT = 'act_complement_init_alt'

    # 激活反转自适应更新方案的类型
    ACT_COMPLEMENT_ADAPT_KRT = 'act_complement_adapt_krt'
    ACT_COMPLEMENT_ADAPT_STB = 'act_complement_adapt_stb'

    # 仿射参数的模式
    ZETA_MODE_CONST = 'zeta_mode_const'
    ZETA_MODE_PARAM = 'zeta_mode_param'
    ZETA_MODE_VEC = 'zeta_mode_vec'
    ZETA_MODE_MAT = 'zeta_mode_mat'

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 weight_init=None,  # 权重初始化方法
                 weight_init_nrm=False,  # 是否对权重进行归一化
                 weight_zeta_mode=None,  # 权重仿射参数模式
                 weight_zeta=0.,  # 权重仿射参数值
                 lrn_sim=kernel_mult2d,  # 学习相似性函数
                 lrn_act=identity,  # 学习激活函数
                 lrn_cmp=False,  # 学习比较函数
                 lrn_t=False,  # 学习时间
                 lrn_bias=False,  # 学习偏置
                 out_sim=kernel_mult2d,  # 输出相似性函数
                 out_act=identity,  # 输出激活函数
                 out_cmp=False,  # 输出比较函数
                 out_t=False,  # 输出时间
                 out_bias=False,  # 输出偏置
                 competitive=None,  # 竞争激活函数
                 act_complement_init=None,  # 激活互补初始化方式
                 act_complement_ratio=0,  # 激活互补比例
                 act_complement_adapt=None,  # 激活互补自适应更新方案
                 act_complement_grp=False,  # 激活互补组
                 act_complement_affine=False,  # 激活互补仿射变换
                 gating=GATE_HEBB,  # 门控类型
                 upd_rule=UPD_RECONSTR,  # 更新规则
                 y_prime_gating=False,  # 是否对 y' 进行门控
                 z_prime_gating=False,  # 是否对 z' 进行门控
                 reconstruction=REC_LIN_CMB,  # 重建方式
                 reduction=RED_AVG,  # 更新缩减方式
                 bias_init=None,  # 偏置初始化
                 bias_mode=None,  # 偏置模式
                 bias_agg=False,  # 偏置聚合
                 bias_target=0,  # 偏置目标
                 bias_gating=None,  # 偏置门控
                 bias_var_gating=False,  # 偏置方差门控
                 bias_zeta_mode=None,  # 偏置仿射参数模式
                 bias_zeta=0.,  # 偏置仿射参数值
                 var_adaptive=False,  # 自适应方差
                 var_affine=False,  # 方差仿射
                 conserve_var=True,  # 保持方差
                 alpha_l=1,  # 局部更新的权重
                 alpha_g=0,  # 全局更新的权重
                 alpha_bias_l=1,  # 局部偏置更新的权重
                 alpha_bias_g=0,  # 全局偏置更新的权重
                 beta=.1):  # 时间常数，用于运行统计跟踪
        super(HebbianConv2d, self).__init__()

        # 初始化权重
        if hasattr(kernel_size, '__len__') and len(kernel_size) == 1:
            kernel_size = kernel_size[0]
        if not hasattr(kernel_size, '__len__'):
            kernel_size = [kernel_size, kernel_size]

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]),
            requires_grad=True
        )

        if not callable(weight_init) and weight_init is not None:
            raise ValueError("Argument weight_init must be callable or None")
        if weight_init is None:
            weight_init = weight_init_std  # 默认初始化
        self.weight = weight_init(self.weight)

        if weight_zeta_mode not in [None, self.ZETA_MODE_CONST, self.ZETA_MODE_PARAM, self.ZETA_MODE_VEC,
                                    self.ZETA_MODE_MAT]:
            raise ValueError("Invalid value for argument weight_zeta_mode: " + str(weight_zeta_mode))

        self.weight_zeta_mode = weight_zeta_mode
        if self.weight_zeta_mode is not None:
            self.weight_ada = nn.Parameter(
                torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]),
                requires_grad=True
            )
            self.weight_ada = weight_init(self.weight_ada)
            self.weight_zeta = weight_zeta
            if self.weight_zeta_mode == self.ZETA_MODE_PARAM:
                self.weight_zeta = nn.Parameter(torch.tensor(weight_zeta).float(), requires_grad=True)
            if self.weight_zeta_mode == self.ZETA_MODE_VEC:
                self.weight_zeta = nn.Parameter(weight_zeta * torch.ones(out_channels).float().view(-1, 1, 1, 1),
                                                requires_grad=True)
            if self.weight_zeta_mode == self.ZETA_MODE_MAT:
                self.weight_zeta = nn.Parameter(weight_zeta * torch.ones_like(self.weight_ada).float(),
                                                requires_grad=True)
            self.weight = self.weight / (1 + (1 + self.weight_zeta))
            self.weight_ada = self.weight_ada / (1 + (1 + self.weight_zeta))
        else:
            self.register_parameter('weight_ada', None)
            self.register_parameter('weight_zeta', None)

        if weight_init_nrm:  # 对权重进行归一化
            w_norm = self.get_weight().view(self.weight.size(0), -1).norm(dim=1, p=2).view(-1, 1, 1, 1)
            self.weight = self.weight / w_norm
            if self.weight_ada is not None:
                self.weight_ada = self.weight_ada / w_norm

        # Alpha 是决定全局和局部更新之间权衡的常数
        self.alpha_l = alpha_l
        self.alpha_g = alpha_g
        self.alpha_bias_l = alpha_bias_l
        self.alpha_bias_g = alpha_bias_g

        # 设置相似性和激活函数
        self.lrn_sim = lrn_sim
        self.lrn_act = lrn_act
        self.lrn_cmp = lrn_cmp
        self.lrn_t = lrn_t
        self.lrn_bias = lrn_bias
        self.out_sim = out_sim
        self.out_act = out_act
        self.out_cmp = out_cmp
        self.out_t = out_t
        self.out_bias = out_bias
        self.competitive = competitive  # 如果为 None 则为恒等映射
        if self.competitive.out_channels is not None and self.competitive.out_channels != out_channels:
            raise ValueError("Argument out_channels: " + str(out_channels) + " and competitive.out_channels: " + str(
                self.competitive.out_channels) + " must match")

        if act_complement_init not in [None, self.ACT_COMPLEMENT_INIT_RAND, self.ACT_COMPLEMENT_INIT_SPLT,
                                       self.ACT_COMPLEMENT_INIT_ALT]:
            raise ValueError("Invalid value for argument act_complement_init: " + str(act_complement_init))
        if act_complement_adapt not in [None, self.ACT_COMPLEMENT_ADAPT_STB, self.ACT_COMPLEMENT_ADAPT_KRT]:
            raise ValueError("Invalid value for argument act_complement_adapt: " + str(act_complement_adapt))
        if act_complement_ratio > 1.0:
            raise ValueError("Invalid value for argument act_complement_ratio: " + str(
                act_complement_ratio) + " (required float <= 1.0)")

        kappa = None
        self.act_complement_from_idx = out_channels
        if act_complement_init == self.ACT_COMPLEMENT_INIT_RAND:
            kappa = (torch.rand(out_channels) < act_complement_ratio).float()
        if act_complement_init == self.ACT_COMPLEMENT_INIT_SPLT:
            kappa = torch.zeros(out_channels).float()
            self.act_complement_from_idx = out_channels - int(round(act_complement_ratio * out_channels))
            if self.act_complement_from_idx < out_channels:
                kappa[self.act_complement_from_idx:] = 1.
        if act_complement_init == self.ACT_COMPLEMENT_INIT_ALT:
            if act_complement_ratio <= 0.5:
                kappa = torch.zeros(out_channels).float()
                if act_complement_ratio > 0:
                    n = int(round(1 / act_complement_ratio))
                    idx = [n * (i + 1) - 1 for i in range(out_channels // n)]
                    kappa[idx] = 1.
            else:
                kappa = torch.ones(out_channels).float()
                if act_complement_ratio < 1:
                    n = int(round(1 / (1 - act_complement_ratio)))
                    idx = [n * i for i in range(out_channels // n)]
                    kappa[idx] = 0.

        self.register_buffer('kappa', kappa)
        self.act_complement_adapt = act_complement_adapt
        self.register_buffer('m2', torch.ones(out_channels).float() if self.act_complement_adapt is not None else None)
        self.register_buffer('m4',
                             (3 * torch.ones(out_channels).float()) if self.act_complement_adapt is not None else None)
        self.register_buffer('rho', (3 * torch.ones(
            out_channels).float()) if self.act_complement_adapt == self.ACT_COMPLEMENT_ADAPT_STB else None)
        self.act_complement_grp = act_complement_grp and (self.act_complement_from_idx < out_channels) and (
                    self.act_complement_adapt is None)
        self.kappa_affine = act_complement_affine
        if self.kappa_affine:
            self.kappa_trainable = nn.Parameter(torch.zeros(out_channels).float(), requires_grad=True)
        else:
            self.register_parameter('kappa_trainable', None)

        # 初始化偏置
        if bias_mode not in [None, self.BIAS_MODE_BASE, self.BIAS_MODE_TARG, self.BIAS_MODE_STD, self.BIAS_MODE_PERC,
                             self.BIAS_MODE_EXP, self.BIAS_MODE_VALUE]:
            raise ValueError("Invalid value for argument bias_mode: " + str(bias_mode))
        if not (bias_init is None or isinstance(bias_init, int) or isinstance(bias_init, float) or (
                isinstance(bias_init, str) and bias_mode == self.BIAS_MODE_VALUE) or callable(bias_init)):
            raise ValueError(
                "Invalid value for argument bias_init: " + str(bias_init) + " when argument bias_mode is: " + str(
                    bias_mode))
        if bias_gating not in [None, self.GATE_BASE, self.GATE_HEBB, self.GATE_DIFF, self.GATE_SMAX]:
            raise ValueError("Invalid value for argument bias_gating: " + str(bias_gating))

        bias = None
        if isinstance(bias_init, int) or isinstance(bias_init, float) or isinstance(bias_init, str):
            bias = bias_init
        if callable(bias_init):
            bias = bias_init(self.weight)

        self.bias_mode = bias_mode
        if self.bias_mode == self.BIAS_MODE_VALUE:
            self.bias = bias
        else:
            self.bias = nn.Parameter(bias * torch.ones(out_channels).float(),
                                     requires_grad=True) if bias is not None else self.register_parameter('bias', None)

        self.bias_agg = bias_agg
        self.bias_target = bias_target
        self.bias_gating = bias_gating
        self.bias_var_gating = bias_var_gating
        self.using_updatable_bias = self.bias is not None and self.bias_mode != self.BIAS_MODE_VALUE
        self.using_adaptive_bias = self.alpha_bias_l != 0 and self.using_updatable_bias and self.bias_mode is not None and self.lrn_bias

        if bias_zeta_mode not in [None, self.ZETA_MODE_CONST, self.ZETA_MODE_PARAM, self.ZETA_MODE_VEC,
                                  self.ZETA_MODE_MAT]:
            raise ValueError("Invalid value for argument bias_zeta_mode: " + str(bias_zeta_mode))

        self.bias_zeta_mode = bias_zeta_mode
        if self.bias_zeta_mode is not None:
            if not self.using_updatable_bias:
                raise ValueError("Invalid argument bias_zeta_mode when bias in non-numeric")
            self.bias_ada = nn.Parameter(bias * torch.ones(out_channels).float(), requires_grad=True)
            self.bias_zeta = bias_zeta
            if self.bias_zeta_mode == self.ZETA_MODE_PARAM:
                self.bias_zeta = nn.Parameter(torch.tensor(bias_zeta).float(), requires_grad=True)
            if self.bias_zeta_mode in [self.ZETA_MODE_VEC, self.ZETA_MODE_MAT]:
                self.bias_zeta = nn.Parameter(bias_zeta * torch.ones(out_channels).float(), requires_grad=True)
            self.bias = self.bias / (1 + (1 + self.bias_zeta))
            self.bias_ada = self.bias_ada / (1 + (1 + self.bias_zeta))
        else:
            self.register_parameter('bias_ada', None)
            self.register_parameter('bias_zeta', None)

        # 学习规则
        self.teacher_signal = None  # 用于监督训练的教师信号
        if gating not in [self.GATE_BASE, self.GATE_HEBB, self.GATE_DIFF, self.GATE_SMAX]:
            raise ValueError("Invalid value for argument gating: " + str(gating))
        self.gating = gating

        if reconstruction not in [None, self.REC_QNT, self.REC_QNT_SGN, self.REC_LIN_CMB]:
            raise ValueError("Invalid value for argument reconstruction: " + str(reconstruction))
        self.reconstruction = reconstruction

        if upd_rule not in [None, self.UPD_RECONSTR, self.UPD_ICA, self.UPD_HICA, self.UPD_ICA_NRM, self.UPD_HICA_NRM]:
            raise ValueError("Invalid value for argument upd_rule: " + str(upd_rule))
        self.upd_rule = upd_rule
        self.using_adaptive_weight = self.alpha_l != 0 and self.upd_rule is not None
        self.y_prime_gating = y_prime_gating
        self.z_prime_gating = z_prime_gating

        if reduction not in [self.RED_AVG, self.RED_W_AVG]:
            raise ValueError("Invalid value for argument reduction: " + str(reduction))
        self.reduction = reduction

        # 自适应方差归一化
        self.beta = beta  # Beta 是用于运行统计跟踪的时间常数
        self.trk = nn.BatchNorm2d(out_channels, momentum=self.beta, affine=var_affine)
        self.var_adaptive = var_adaptive
        self.conserve_var = conserve_var

        # 存储权重更新的变量
        self.delta_w = None
        self.delta_bias = None

    # 设置教师信号
    def set_teacher_signal(self, t):
        self.teacher_signal = t

    # 前向传播函数
    def forward(self, x):
        # 如果处于训练模式，则计算更新
        if self.training:
            self.compute_update(x)
        # 计算输出
        out = self.out_sim(x, self.get_weight(), self.get_bias() if self.out_bias else None)
        out_shape = out.size()
        # 应用激活函数
        out = self.apply_act(out)
        # 获取教师信号
        t = self.teacher_signal if self.out_t else None
        if t is not None:
            t = t.unsqueeze(2).unsqueeze(3) * torch.ones_like(out)
        # 调整输出形状以便于竞争计算
        out = out.permute(0, 2, 3, 1).contiguous().view(-1, out.size(1))
        if t is not None:
            t = t.permute(0, 2, 3, 1).contiguous().view(-1, self.weight.size(0))
        # 应用竞争激活
        out = self.competitive(out, t) if self.competitive is not None and self.out_cmp else out
        out = out * t if t is not None else out
        # 恢复输出形状
        return out.view(out_shape[0], out_shape[2], out_shape[3], out_shape[1]).permute(0, 3, 1, 2).contiguous()

    # 获取权重
    def get_weight(self):
        weight = self.weight
        if self.weight_ada is not None:
            weight = weight + (1 + self.weight_zeta) * self.weight_ada
        return weight

    # 获取偏置
    def get_bias(self):
        bias = self.bias
        if self.bias_ada is not None:
            bias = bias + (1 + self.bias_zeta) * self.bias_ada
        return bias.mean() if self.bias_agg else self.bias

    # 应用激活函数
    def apply_act(self, s, lrn=False, cpl=True):
        s_bar = s

        # 如果需要，在激活函数前进行归一化
        if lrn:
            _ = self.trk(s)  # 跟踪统计数据
        if self.var_adaptive:
            s_bar = (s - self.trk.running_mean.view(1, -1, 1, 1)) / (
                    (self.trk.running_var.view(1, -1, 1, 1) + self.trk.eps) ** 0.5)
            if self.trk.affine:
                s_bar = s_bar * self.trk.weight.view(1, -1, 1, 1) + self.trk.bias.view(1, -1, 1, 1)
            s_bar = s_bar + self.trk.running_mean.view(1, -1, 1, 1)  # 恢复原始均值，仅归一化方差

        # 应用激活函数
        y = self.lrn_act(s_bar) if lrn else self.out_act(s_bar)
        # 如果需要，应用激活互补
        if cpl:
            kappa = self.kappa.view(1, -1, 1, 1) if self.kappa is not None else 0.
            if self.kappa_affine:
                kappa = kappa + self.kappa_trainable.view(1, -1, 1, 1)
            y = kappa * s_bar - (2 * kappa - 1) * y

        # 如果需要，恢复原始方差信息
        if self.conserve_var and self.var_adaptive:
            y = y * (self.trk.running_var.view(1, -1, 1, 1) + self.trk.eps) ** 0.5

        return y

    # 计算权重和偏置的更新
    def compute_update(self, x):
        # 存储之前的梯度计算状态，并在计算更新前禁用梯度计算
        prev_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

        # 如果使用自适应权重或偏置，或者启用自适应方差或激活互补自适应，则计算激活状态
        if self.using_adaptive_weight or self.using_adaptive_bias or self.var_adaptive or self.act_complement_adapt is not None:
            # 计算层的激活状态：s, y, y'
            s = self.lrn_sim(x, self.get_weight(),
                             self.get_bias() if self.lrn_bias else None)  # 计算输入和权重之间的相似性度量

            # 计算 y 和 y'（如果需要 y' 门控）
            if self.y_prime_gating:
                torch.set_grad_enabled(True)  # 启用梯度计算以计算 y'
                s.requires_grad = True
            y = self.apply_act(s, lrn=True)
            y_prime = torch.ones_like(s)
            if self.y_prime_gating:
                y.backward(torch.ones_like(y), retain_graph=prev_grad_enabled, create_graph=prev_grad_enabled)
                y_prime = s.grad
                s.grad = None
                torch.set_grad_enabled(False)

            # 跟踪高阶矩
            if self.act_complement_adapt == self.ACT_COMPLEMENT_ADAPT_KRT:
                # 更新统计数据并确定 kappa
                self.m2 = (1 - self.beta) * self.m2 + self.beta * s.pow(2).mean(dim=(0, 2, 3))
                self.m4 = (1 - self.beta) * self.m4 + self.beta * s.pow(4).mean(dim=(0, 2, 3))
                self.kappa = ((self.m4 - 3 * self.m2 ** 2) < 0).float()
            if self.act_complement_adapt == self.ACT_COMPLEMENT_ADAPT_STB:
                # 计算未互补的 y'，这对这种自适应很重要
                torch.set_grad_enabled(True)  # 启用梯度计算以计算未互补的 y'
                s.requires_grad = True
                y_uncpl = self.apply_act(s, cpl=False)
                y_uncpl.backward(torch.ones_like(y_uncpl), retain_graph=prev_grad_enabled,
                                 create_graph=prev_grad_enabled)
                y_uncpl_prime = s.grad
                s.grad = None
                torch.set_grad_enabled(False)
                # 更新统计数据并确定 kappa
                self.m2 = (1 - self.beta) * self.m2 + self.beta * s.pow(2).mean(dim=(0, 2, 3))
                self.m4 = (1 - self.beta) * self.m4 + self.beta * (s * y_uncpl).mean(dim=(0, 2, 3))
                self.rho = (1 - self.beta) * self.rho + self.beta * y_uncpl_prime.mean(dim=(0, 2, 3))
                self.kappa = ((self.m4 - self.m2 * self.rho) < 0).float()

            if self.using_adaptive_weight or self.using_adaptive_bias:
                # 准备必要的张量，并将它们设置为正确的形状
                t = self.teacher_signal if self.lrn_t else None
                if t is not None:
                    t = t.unsqueeze(2).unsqueeze(3) * torch.ones_like(s)
                s = s.permute(0, 2, 3, 1).contiguous().view(-1, self.weight.size(0))
                y = y.permute(0, 2, 3, 1).contiguous().view(-1, self.weight.size(0))
                y_prime = y_prime.permute(0, 2, 3, 1).contiguous().view(-1, self.weight.size(0))
                if t is not None:
                    t = t.permute(0, 2, 3, 1).contiguous().view(-1, self.weight.size(0))
                x_unf = unfold_map2d(x, self.weight.size(2), self.weight.size(3))
                x_unf = x_unf.permute(0, 2, 3, 1, 4).contiguous().view(s.size(0), 1, -1)

                # 运行竞争，如果需要，也计算竞争非线性的导数
                if self.z_prime_gating:
                    torch.set_grad_enabled(True)  # 启用梯度计算以计算 z'
                    s.requires_grad = True
                z = self.competitive(y, t, lrn=True) if self.competitive is not None and self.lrn_cmp else y
                z_prime = torch.ones_like(y)
                if self.z_prime_gating:
                    z.backward(torch.ones_like(z), retain_graph=prev_grad_enabled, create_graph=prev_grad_enabled)
                    z_prime = y.grad
                    y.grad = None
                    torch.set_grad_enabled(False)

                # 如果需要，使用教师信号进行门控
                zt = z * t if t is not None else z

                if self.using_adaptive_weight:
                    # 计算步调调制系数
                    r = zt  # GATE_BASE
                    if self.gating == self.GATE_HEBB:
                        r = r * y
                    if self.gating == self.GATE_DIFF:
                        r = r - y
                    if self.gating == self.GATE_SMAX:
                        r = r - torch.softmax(y, dim=1)
                    if self.y_prime_gating:
                        r = y_prime * r
                    if self.z_prime_gating:
                        r = z_prime * r

                    # 计算批次中更新缩减/聚合的系数
                    # 在批处理输入数据时，我们需要将每个卷积核的不同更新步骤聚合为一个独特的更新。
                    # 这是因为在使用批次输入时，每个卷积核可能会接收到多个更新步骤。
                    # 我们通过计算更新步骤的加权平均（RED_W_AVG）或非加权平均（RED_AVG）来实现聚合。
                    # Compute the coefficients for update reduction/aggregation over the batch.
                    # Since we use batches of inputs, we need to aggregate the different update steps of each kernel in a unique
                    # update. We do this by taking the weighted average of the steps, the weights being the r coefficients that
                    # determine the length of each step (RED_W_AVG), or the unweighted average (RED_AVG).
                    c = 1 / r.size(0)
                    if self.reduction == self.RED_W_AVG:
                        r_sum = r.abs().sum(dim=0, keepdim=True)
                        r_sum = r_sum + (r_sum == 0).float()  # 防止除以零
                        c = r.abs() / r_sum

                    # 计算 delta_w
                    delta_w_agg = torch.zeros_like(self.weight.view(self.weight.size(0), -1))
                    for grp in range(2):  # 重复计算使用互补非线性的两个神经元组
                        if grp == 1 and not self.act_complement_grp:
                            break
                        grp_slice = slice(self.weight.size(0))
                        if self.act_complement_grp:
                            grp_slice = slice(self.act_complement_from_idx) if grp == 0 else slice(
                                self.act_complement_from_idx,
                                self.weight.size(0))
                        w = (self.weight_ada if self.weight_ada is not None else self.weight).view(1,
                                                                                                   self.weight.size(0),
                                                                                                   -1)[:, grp_slice, :]
                        x_bar = None
                        rrlw = None
                        sw = None

                        if P.HEBB_FASTHEBB:
                            if not P.HEBB_REORDMULT:
                                if self.upd_rule == self.UPD_ICA or self.upd_rule == self.UPD_ICA_NRM:  # ICA的 s^T * W 计算
                                    for i in range((w.size(1) // P.HEBB_UPD_GRP) + (
                                    1 if w.size(1) % P.HEBB_UPD_GRP != 0 else 0)):
                                        start = i * P.HEBB_UPD_GRP
                                        end = min((i + 1) * P.HEBB_UPD_GRP, w.size(1))
                                        w_i = w[:, start:end, :]
                                        s_i = s[:, grp_slice].unsqueeze(2)[:, start:end, :]
                                        r_i = r[:, grp_slice].unsqueeze(2)[:, start:end, :]
                                        sw = (r_i * s_i).view(r_i.size(0), -1).matmul(
                                            w_i.view(w_i.size(1), -1)).unsqueeze(1) + (sw if sw is not None else 0.)

                            for i in range(
                                    (w.size(1) // P.HEBB_UPD_GRP) + (1 if w.size(1) % P.HEBB_UPD_GRP != 0 else 0)):
                                start = i * P.HEBB_UPD_GRP
                                end = min((i + 1) * P.HEBB_UPD_GRP, w.size(1))
                                w_i = w[:, start:end, :]
                                s_i = s[:, grp_slice].unsqueeze(2)[:, start:end, :]
                                y_i = y[:, grp_slice].unsqueeze(2)[:, start:end, :]
                                r_i = r[:, grp_slice].unsqueeze(2)[:, start:end, :]
                                c_i = c[:, grp_slice].unsqueeze(2)[:, start:end, :] if isinstance(c,
                                                                                                  torch.Tensor) else c
                                rc = r_i * c_i

                                # 计算更新步调
                                delta_w_i = torch.zeros_like(w_i)
                                if self.upd_rule == self.UPD_RECONSTR:
                                    delta_w_i = rc.view(rc.size(0), -1).t().matmul(x_unf.view(x_unf.size(0), -1))
                                    # 根据重建类型计算重建
                                    if self.reconstruction == self.REC_QNT:
                                        delta_w_i = delta_w_i - rc.sum(0) * w_i.view(w_i.size(1), -1)
                                    elif self.reconstruction == self.REC_QNT_SGN:
                                        delta_w_i = delta_w_i - (rc * r_i.sign()).sum(0) * w_i.view(w_i.size(1), -1)
                                    elif self.reconstruction == self.REC_LIN_CMB:
                                        if P.HEBB_REORDMULT:
                                            l_i = (torch.arange(w.size(1), device=w.device).unsqueeze(0).repeat(
                                                w_i.size(1), 1) <= torch.arange(start, end, device=w.device).unsqueeze(
                                                1)).float()
                                            rrlw = (rc.view(rc.size(0), -1).t().matmul(r[:, grp_slice]) * l_i).matmul(
                                                w.view(w.size(1), -1))
                                            delta_w_i = delta_w_i - rrlw
                                        else:
                                            x_bar = torch.cumsum(r_i * w_i, dim=1) + (
                                                x_bar[:, -1, :].unsqueeze(1) if x_bar is not None else 0.)
                                            delta_w_i = delta_w_i - rc.permute(1, 2, 0).matmul(
                                                x_bar.permute(1, 0, 2)).view(w_i.size(1), -1)
                                if self.upd_rule in [self.UPD_ICA, self.UPD_HICA, self.UPD_ICA_NRM, self.UPD_HICA_NRM]:
                                    if P.HEBB_REORDMULT:
                                        if self.upd_rule == self.UPD_HICA or self.upd_rule == self.UPD_HICA_NRM:
                                            l_i = (torch.arange(w.size(1), device=w.device).unsqueeze(0).repeat(
                                                w_i.size(1), 1) <= torch.arange(start, end, device=w.device).unsqueeze(
                                                1)).float()
                                            rysw = ((rc * y_i).view(rc.size(0), -1).t().matmul(
                                                r[:, grp_slice] * s[:, grp_slice]) * l_i).matmul(w.view(w.size(1), -1))
                                        else:
                                            rysw = (rc * y_i).view(rc.size(0), -1).t().matmul(
                                                r[:, grp_slice] * s[:, grp_slice]).matmul(w.view(w.size(1), -1))
                                    else:
                                        if self.upd_rule == self.UPD_HICA or self.upd_rule == self.UPD_HICA_NRM:
                                            sw = torch.cumsum((r_i * s_i) * w_i, dim=1) + (
                                                sw[:, -1, :].unsqueeze(1) if sw is not None else 0.)
                                        rysw = (rc * y_i * sw).sum(0)
                                    if self.upd_rule == self.UPD_ICA or self.upd_rule == self.UPD_HICA:
                                        delta_w_i = rc.sum(0) * w_i.view(w_i.size(1), -1) - rysw
                                    if self.upd_rule == self.UPD_ICA_NRM or self.upd_rule == self.UPD_HICA_NRM:
                                        delta_w_i = (rysw * w_i.view(w_i.size(1), -1)).sum(dim=1,
                                                                                           keepdim=True) * w_i.view(
                                            w_i.size(1), -1) - rysw

                                # 将聚合的更新存储在缓冲区中
                                delta_w_agg[grp_slice, :][start:end, :] = delta_w_i

                        else:
                            if self.upd_rule == self.UPD_ICA or self.upd_rule == self.UPD_ICA_NRM:  # ICA的 s^T * W 计算
                                for i in range(
                                        (w.size(1) // P.HEBB_UPD_GRP) + (1 if w.size(1) % P.HEBB_UPD_GRP != 0 else 0)):
                                    start = i * P.HEBB_UPD_GRP
                                    end = min((i + 1) * P.HEBB_UPD_GRP, w.size(1))
                                    w_i = w[:, start:end, :]
                                    s_i = s[:, grp_slice].unsqueeze(2)[:, start:end, :]
                                    r_i = r[:, grp_slice].unsqueeze(2)[:, start:end, :]
                                    sw = (r_i * s_i * w_i).sum(dim=1, keepdim=True) + (sw if sw is not None else 0.)

                            for i in range(
                                    (w.size(1) // P.HEBB_UPD_GRP) + (1 if w.size(1) % P.HEBB_UPD_GRP != 0 else 0)):
                                start = i * P.HEBB_UPD_GRP
                                end = min((i + 1) * P.HEBB_UPD_GRP, w.size(1))
                                w_i = w[:, start:end, :]
                                s_i = s[:, grp_slice].unsqueeze(2)[:, start:end, :]
                                y_i = y[:, grp_slice].unsqueeze(2)[:, start:end, :]
                                r_i = r[:, grp_slice].unsqueeze(2)[:, start:end, :]
                                c_i = c[:, grp_slice].unsqueeze(2)[:, start:end, :] if isinstance(c,
                                                                                                  torch.Tensor) else c

                                # 计算更新步调
                                delta_w_i = torch.zeros_like(w_i)
                                if self.upd_rule == self.UPD_RECONSTR:
                                    # 根据重建类型计算重建
                                    if self.reconstruction == self.REC_QNT:
                                        x_bar = w_i
                                    elif self.reconstruction == self.REC_QNT_SGN:
                                        x_bar = r_i.sign() * w_i
                                    elif self.reconstruction == self.REC_LIN_CMB:
                                        x_bar = torch.cumsum(r_i * w_i, dim=1) + (
                                            x_bar[:, -1, :].unsqueeze(1) if x_bar is not None else 0.)
                                    else:
                                        x_bar = 0.
                                    delta_w_i = r_i * (x_unf - x_bar)
                                if self.upd_rule in [self.UPD_ICA, self.UPD_HICA, self.UPD_ICA_NRM, self.UPD_HICA_NRM]:
                                    if self.upd_rule == self.UPD_HICA或self.UPD_HICA_NRM:
                                        sw = torch.cumsum((r_i * s_i) * w_i, dim=1) + (
                                            sw[:, -1, :].unsqueeze(1) if sw is not None else 0.)
                                    ysw = (y_i * sw)
                                    if self.upd_rule == self.UPD_ICA or self.UPD_HICA:
                                        delta_w_i = r_i * (w_i - ysw)
                                    if self.UPD_ICA_NRM or self.UPD_HICA_NRM:
                                        delta_w_i = r_i * ((ysw * w_i).sum(dim=2, keepdim=True) * w_i - ysw)

                                # 聚合批次中的更新
                                delta_w_agg[grp_slice, :][start:end, :] = (delta_w_i * c_i).sum(0)

                    # 存储 delta
                    self.delta_w = delta_w_agg.view_as(self.weight)

                if self.using_adaptive_bias:
                    # 计算步调调制系数
                    r = zt  # GATE_BASE
                    b = (self.bias_ada if self.bias_ada is not None else self.bias).view(1, -1)
                    if self.bias_gating == self.GATE_HEBB:
                        r = r * y
                    if self.bias_gating == self.GATE_DIFF:
                        r = r - y
                    if self.bias_gating == self.GATE_SMAX:
                        r = r - torch.softmax(y, dim=1)
                    if self.bias_gating is None:
                        r = 1.
                    if self.y_prime_gating:
                        r = y_prime * r
                    if self.z_prime_gating:
                        r = z_prime * r
                    if self.bias_var_gating:
                        r = r * s * (torch.log(s) / b.view(1, -1))

                    # 计算批次中更新缩减/聚合的系数
                    c = 1 / r.size(0)
                    if self.reduction == self.RED_W_AVG:
                        r_sum = r.abs().sum(dim=0, keepdim=True)
                        r_sum = r_sum + (r_sum == 0).float()  # 防止除以零
                        c = r.abs() / r_sum

                    # 计算 Delta 偏置
                    delta_bias = torch.zeros_like(self.bias).unsqueeze(0)
                    if self.bias_mode == self.BIAS_MODE_BASE:
                        delta_bias = r
                    if self.bias_mode == self.BIAS_MODE_TARG:
                        # 注意：self.bias_target 是我们希望偏置达到的目标均值
                        delta_bias = r * (self.bias_target - s)
                    if self.bias_mode == self.BIAS_MODE_STD:
                        # 注意：self.bias_target 是我们希望偏置达到的标准差倍数
                        delta_bias = r * (self.bias_target * (self.trk.running_var ** 0.5).view(1, -1) - s)
                    if self.bias_mode == self.BIAS_MODE_PERC:
                        # 注意：self.bias_target 是我们希望相似性函数达到的零以上样本的目标百分位数
                        delta_bias = r * (self.bias_target - (s < 0).float())
                    if self.bias_mode == self.BIAS_MODE_EXP:
                        # 注意：self.bias_target 是我们希望相似性函数尺度达到的标准差倍数
                        delta_bias = r * b * (1 - (-torch.log(s)) * self.bias_target)

                    # 聚合批次中的更新并存储 delta
                    self.delta_bias = c.t().unsqueeze(1).matmul(delta_bias.t().unsqueeze(2)).view_as(self.bias)

        # 恢复梯度计算
        torch.set_grad_enabled(prev_grad_enabled)

    # 从 self.delta_w 和 self.delta_bias 中获取局部更新，从 self.weight.grad 和 self.bias.grad 中获取全局更新，并使用参数 alpha 将它们组合起来
    def local_update(self):
        if self.delta_w is not None or self.weight.grad is not None:
            if self.weight_ada is None:
                # 注意：self.delta_w 前有负号，因为优化器将向相反方向更新
                self.weight.grad = self.alpha_l * (-self.delta_w if self.delta_w is not None else torch.zeros_like(
                    self.weight).float()) + self.alpha_g * (
                                       self.weight.grad if self.weight.grad is not None else torch.zeros_like(
                                           self.weight).float())
            else:
                self.weight_ada.grad = self.alpha_l * (
                    -self.delta_w if self.delta_w is not None else torch.zeros_like(self.weight).float())
                self.weight.grad = self.alpha_g * (
                    self.weight.grad if self.weight.grad is not None else torch.zeros_like(self.weight).float())
            self.delta_w = None
        if self.using_updatable_bias and (self.delta_bias is not None or self.bias.grad is not None):
            if self.bias_ada is None:
                # 注意：self.delta_bias 前有负号，因为优化器将向相反方向更新
                self.bias.grad = self.alpha_bias_l * (
                    -self.delta_bias if self.delta_bias is not None else torch.zeros_like(
                        self.bias).float()) + self.alpha_bias_g * (
                                     self.bias.grad if self.bias.grad is not None else torch.zeros_like(
                                         self.bias).float())
            else:
                self.bias_ada.grad = self.alpha_bias_l * (
                    -self.delta_bias if self.delta_bias is not None else torch.zeros_like(self.bias).float())
                self.bias.grad = self.alpha_bias_g * (
                    self.bias.grad if self.bias.grad is not None else torch.zeros_like(self.bias).float())
            self.delta_bias = None
