from neurolab import params as P
import params as PP
from .meta import *

config_base_hebb = {}
gdes_fc_on_hebb_layer_ft = {}
config_base_gdes = {}

for ds in datasets:  # 遍历所有数据集【hebb】
    for da in da_strategies:  # 遍历所有数据增强策略
        for lrn_rule in lrn_rules:  # 遍历所有学习规则
            config_base_hebb[lrn_rule + '_' + ds + da_names[da]] = {
                P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',  # 实验类
                P.KEY_NET_MODULES: 'models.hebb.model_' + str(num_layers[ds]) + 'l.Net',  # 网络模块
                P.KEY_NET_OUTPUTS: net_outputs[ds],  # 网络输出
                P.KEY_DATA_MANAGER: data_managers[ds],  # 数据管理器
                P.KEY_AUGMENT_MANAGER: da_managers[da],  # 数据增强管理器
                P.KEY_AUGM_BEFORE_STATS: True,  # 数据增强在统计之前
                P.KEY_AUGM_STAT_PASSES: da_mult[da],  # 数据增强统计通过次数
                P.KEY_WHITEN: None if lrn_rule_keys[lrn_rule] != 'hwta' else 2,  # 是否使用白化
                P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],  # 训练样本总数
                P.KEY_BATCHSIZE: batch_sizes[ds],  # 批量大小
                P.KEY_INPUT_SHAPE: input_shapes[ds],  # 输入形状
                P.KEY_NUM_EPOCHS: 1,  # 训练轮数
                P.KEY_OPTIM_MANAGER: 'neurolab.optimization.optim.SGDOptimManager',  # 优化管理器
                P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager',
                                            'neurolab.optimization.metric.TopKAccMetricManager'],  # 评估指标管理器
                P.KEY_TOPKACC_K: [1, 5],  # Top-K 精度
                P.KEY_LEARNING_RATE: hebb_lrn_rates[lrn_rule_keys[lrn_rule]][ds],  # 学习率
                P.KEY_LOCAL_LRN_RULE: lrn_rule_keys[lrn_rule],  # 局部学习规则
                PP.KEY_COMPETITIVE_ACT: lrn_rule_competitive_act[lrn_rule],  # 竞争激活
                PP.KEY_COMPETITIVE_K: lrn_rule_k[lrn_rule],  # 竞争 K 值
                P.KEY_DEEP_TEACHER_SIGNAL: lrn_rule_dts[lrn_rule],  # 深度教师信号
            }

            gdes_fc_on_hebb_layer_ft[lrn_rule + '_' + ds + da_names[da]] = {
                P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',  # 实验类型
                P.KEY_NET_MODULES: ['models.hebb.model_' + str(num_layers[ds]) + 'l.Net', 'models.gdes.fc.Net'],  # 网络模块
                P.KEY_NET_MDL_PATHS: [  # 网络模型路径
                    P.PROJECT_ROOT + '/results/configs/HuaCL/pretrain/config_base_hebb[pswta_cifar10]/iter0/models/model0.pt'],
                P.KEY_NET_OUTPUTS: ['bn5', 'fc'],  # 网络输出
                P.KEY_DATA_MANAGER: data_managers[ds],  # 数据管理器
                P.KEY_AUGMENT_MANAGER: da_managers[da],  # 数据增强管理器
                P.KEY_AUGM_BEFORE_STATS: True,  # 数据增强在统计之前
                P.KEY_AUGM_STAT_PASSES: da_mult[da],  # 数据增强统计通过次数
                P.KEY_WHITEN: None if lrn_rule_keys[lrn_rule] != 'hwta' else 2,  # 是否使用白化
                P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],  # 训练样本总数
                P.KEY_BATCHSIZE: batch_sizes[ds],  # 批量大小
                P.KEY_INPUT_SHAPE: input_shapes[ds],  # 输入形状
                P.KEY_NUM_EPOCHS: da_mult[da] * 20,  # 训练轮数
                P.KEY_OPTIM_MANAGER: 'neurolab.optimization.optim.SGDOptimManager',  # 优化管理器
                P.KEY_SCHED_MANAGER: 'neurolab.optimization.sched.MultiStepSchedManager',  # 调度管理器
                P.KEY_LOSS_METRIC_MANAGER: 'neurolab.optimization.metric.CrossEntMetricManager',  # 损失度量管理器
                P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager',  # 评估指标管理器
                                            'neurolab.optimization.metric.TopKAccMetricManager'],
                P.KEY_TOPKACC_K: [1, 5],  # Top-K 精度
                P.KEY_LEARNING_RATE: 1e-3,  # 学习率
                P.KEY_ALPHA_L: 0, P.KEY_ALPHA_G: 1,  # Alpha参数
                P.KEY_LR_DECAY: 0.5 if da == 'no_da' else 0.1,  # 学习率衰减
                P.KEY_MILESTONES: range(10, 20) if da == 'no_da' else [20, 30] if da == 'light_da' else [40, 70,
                                                                                                         90],  # 里程碑
                P.KEY_MOMENTUM: 0.9,  # 动量
                P.KEY_L2_PENALTY: l2_penalties[ds + da_names[da]],  # L2惩罚
                P.KEY_DROPOUT_P: 0.5,  # Dropout参数
                P.KEY_LOCAL_LRN_RULE: lrn_rule_keys[lrn_rule],  # 局部学习规则
                PP.KEY_COMPETITIVE_ACT: lrn_rule_competitive_act[lrn_rule],  # 竞争激活
                PP.KEY_COMPETITIVE_K: lrn_rule_k[lrn_rule],  # 竞争 K 值
                P.KEY_DEEP_TEACHER_SIGNAL: lrn_rule_dts[lrn_rule],  # 深度教师信号
            }

for ds in datasets:
    for da in da_strategies:
        config_base_gdes[ds + da_names[da]] = {
            P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
            P.KEY_NET_MODULES: 'models.gdes.model_' + str(num_layers[ds]) + 'l.Net',
            P.KEY_NET_OUTPUTS: net_outputs[ds],
            P.KEY_DATA_MANAGER: data_managers[ds],
            P.KEY_AUGMENT_MANAGER: da_managers[da],
            P.KEY_AUGM_BEFORE_STATS: True,
            P.KEY_AUGM_STAT_PASSES: da_mult[da],
            P.KEY_WHITEN: None,
            P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
            P.KEY_BATCHSIZE: batch_sizes[ds],
            P.KEY_INPUT_SHAPE: input_shapes[ds],
            P.KEY_NUM_EPOCHS: da_mult[da] * 20,
            P.KEY_OPTIM_MANAGER: 'neurolab.optimization.optim.SGDOptimManager',
            P.KEY_SCHED_MANAGER: 'neurolab.optimization.sched.MultiStepSchedManager',
            P.KEY_LOSS_METRIC_MANAGER: 'neurolab.optimization.metric.CrossEntMetricManager',
            P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager',
                                        'neurolab.optimization.metric.TopKAccMetricManager'],
            P.KEY_TOPKACC_K: [1, 5],
            P.KEY_LEARNING_RATE: 1e-3,
            P.KEY_LR_DECAY: 0.5 if da == 'no_da' else 0.1,
            P.KEY_MILESTONES: range(10, 20) if da == 'no_da' else [20, 30] if da == 'light_da' else [40, 70, 90],
            P.KEY_MOMENTUM: 0.9,
            P.KEY_L2_PENALTY: l2_penalties[ds + da_names[da]],
            P.KEY_DROPOUT_P: 0.5,
        }
