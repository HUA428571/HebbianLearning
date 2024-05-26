from neurolab import params as P
import params as PP
from .meta import *


config_base_hebb = {}
gdes_fc_on_hebb_layer = {}
gdes_fc_on_hebb_layer_ft = {}
hebb_fc_on_hebb_layer = {}
hebb_fc_on_hebb_layer_ft = {}
gdes_fc2_on_hebb_layer = {}
gdes_fc2_on_hebb_layer_ft = {}
hebb_fc2_on_hebb_layer = {}
hebb_fc2_on_hebb_layer_ft = {}
svm_on_hebb_layer = {}
knn_on_hebb_layer = {}
prec_on_hebb_layer = {}
svm_on_hebb_layer_ft = {}
knn_on_hebb_layer_ft = {}
prec_on_hebb_layer_ft = {}


for ds in datasets:  # 遍历所有数据集
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
				P.KEY_TOPKACC_K: [1],  # Top-K 精度
				P.KEY_LEARNING_RATE: hebb_lrn_rates[lrn_rule_keys[lrn_rule]][ds],  # 学习率
				P.KEY_LOCAL_LRN_RULE: lrn_rule_keys[lrn_rule],  # 局部学习规则
				PP.KEY_COMPETITIVE_ACT: lrn_rule_competitive_act[lrn_rule],  # 竞争激活
				PP.KEY_COMPETITIVE_K: lrn_rule_k[lrn_rule],  # 竞争 K 值
				P.KEY_DEEP_TEACHER_SIGNAL: lrn_rule_dts[lrn_rule],  # 深度教师信号
			}
		
			for l in range(1, num_layers[ds]):
				gdes_fc_on_hebb_layer[str(l) + '_' + lrn_rule + '_' + ds + da_names[da]] = {
					P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
					P.KEY_NET_MODULES: 'models.gdes.fc.Net',
					P.KEY_NET_OUTPUTS: 'fc',
					P.KEY_DATA_MANAGER: data_managers[ds],
					P.KEY_AUGMENT_MANAGER: da_managers[da],
					P.KEY_AUGM_BEFORE_STATS: True,
					P.KEY_AUGM_STAT_PASSES: da_mult[da],
					P.KEY_WHITEN: None if lrn_rule_keys[lrn_rule] != 'hwta' else 2,
					P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
					P.KEY_BATCHSIZE: batch_sizes[ds],
					P.KEY_INPUT_SHAPE: input_shapes[ds],
					P.KEY_NUM_EPOCHS: da_mult[da] * 20,
					P.KEY_OPTIM_MANAGER: 'neurolab.optimization.optim.SGDOptimManager',
					P.KEY_SCHED_MANAGER: 'neurolab.optimization.sched.MultiStepSchedManager',
					P.KEY_LOSS_METRIC_MANAGER: 'neurolab.optimization.metric.CrossEntMetricManager',
					P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
					P.KEY_TOPKACC_K: [1, 5],
					P.KEY_LEARNING_RATE: 1e-3,
					P.KEY_LR_DECAY: 0.5 if da == 'no_da' else 0.1,
					P.KEY_MILESTONES: range(10, 20) if da == 'no_da' else [20, 30] if da == 'light_da' else [40, 70, 90],
					P.KEY_MOMENTUM: 0.9,
					P.KEY_L2_PENALTY: 5e-4,
					P.KEY_DROPOUT_P: 0.5,
					P.KEY_LOCAL_LRN_RULE: lrn_rule_keys[lrn_rule],
					PP.KEY_COMPETITIVE_ACT: lrn_rule_competitive_act[lrn_rule],
					PP.KEY_COMPETITIVE_K: lrn_rule_k[lrn_rule],
					P.KEY_PRE_NET_MODULES: ['models.hebb.model_' + str(num_layers[ds]) + 'l.Net'],
					P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/hebb/config_base_hebb[' + lrn_rule + '_' + ds + da_names[da] + ']/iter' + P.STR_TOKEN + '/models/model0.pt'],
					P.KEY_PRE_NET_OUTPUTS: ['bn' + str(l)],
				}

				gdes_fc_on_hebb_layer_ft[str(l) + '_' + lrn_rule + '_' + ds + da_names[da]] = {
					P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
					P.KEY_NET_MODULES: ['models.hebb.model_' + str(num_layers[ds]) + 'l.Net', 'models.gdes.fc.Net'],
					P.KEY_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/hebb/config_base_hebb[' + lrn_rule + '_' + ds + da_names[da] + ']/iter' + P.STR_TOKEN + '/models/model0.pt'],
					P.KEY_NET_OUTPUTS: ['bn' + str(l), 'fc'],
					P.KEY_DATA_MANAGER: data_managers[ds],
					P.KEY_AUGMENT_MANAGER: da_managers[da],
					P.KEY_AUGM_BEFORE_STATS: True,
					P.KEY_AUGM_STAT_PASSES: da_mult[da],
					P.KEY_WHITEN: None if lrn_rule_keys[lrn_rule] != 'hwta' else 2,
					P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
					P.KEY_BATCHSIZE: batch_sizes[ds],
					P.KEY_INPUT_SHAPE: input_shapes[ds],
					P.KEY_NUM_EPOCHS: da_mult[da] * 20,
					P.KEY_OPTIM_MANAGER: 'neurolab.optimization.optim.SGDOptimManager',
					P.KEY_SCHED_MANAGER: 'neurolab.optimization.sched.MultiStepSchedManager',
					P.KEY_LOSS_METRIC_MANAGER: 'neurolab.optimization.metric.CrossEntMetricManager',
					P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
					P.KEY_TOPKACC_K: [1, 5],
					P.KEY_LEARNING_RATE: 1e-3,
					P.KEY_ALPHA_L: 0, P.KEY_ALPHA_G: 1,
					P.KEY_LR_DECAY: 0.5 if da == 'no_da' else 0.1,
					P.KEY_MILESTONES: range(10, 20) if da == 'no_da' else [20, 30] if da == 'light_da' else [40, 70, 90],
					P.KEY_MOMENTUM: 0.9,
					P.KEY_L2_PENALTY: l2_penalties[ds + da_names[da]],
					P.KEY_DROPOUT_P: 0.5,
					P.KEY_LOCAL_LRN_RULE: lrn_rule_keys[lrn_rule],
					PP.KEY_COMPETITIVE_ACT: lrn_rule_competitive_act[lrn_rule],
					PP.KEY_COMPETITIVE_K: lrn_rule_k[lrn_rule],
					P.KEY_DEEP_TEACHER_SIGNAL: lrn_rule_dts[lrn_rule],
				}

				hebb_fc_on_hebb_layer[str(l) + '_' + lrn_rule + '_' + ds + da_names[da]] = {
					P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
					P.KEY_NET_MODULES: 'models.hebb.fc.Net',
					P.KEY_NET_OUTPUTS: 'fc',
					P.KEY_DATA_MANAGER: data_managers[ds],
					P.KEY_AUGMENT_MANAGER: da_managers[da],
					P.KEY_AUGM_BEFORE_STATS: True,
					P.KEY_AUGM_STAT_PASSES: da_mult[da],
					P.KEY_WHITEN: None if lrn_rule_keys[lrn_rule] != 'hwta' else 2,
					P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
					P.KEY_BATCHSIZE: batch_sizes[ds],
					P.KEY_INPUT_SHAPE: input_shapes[ds],
					P.KEY_NUM_EPOCHS: 20,
					P.KEY_OPTIM_MANAGER: 'neurolab.optimization.optim.SGDOptimManager',
					P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
					P.KEY_TOPKACC_K: [1, 5],
					P.KEY_LEARNING_RATE: 1e-3,
					P.KEY_LOCAL_LRN_RULE: lrn_rule_keys[lrn_rule],
					PP.KEY_COMPETITIVE_ACT: lrn_rule_competitive_act[lrn_rule],
					PP.KEY_COMPETITIVE_K: lrn_rule_k[lrn_rule],
					P.KEY_PRE_NET_MODULES: ['models.hebb.model_' + str(num_layers[ds]) + 'l.Net'],
					P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/hebb/config_base_hebb[' + lrn_rule + '_' + ds + da_names[da] + ']/iter' + P.STR_TOKEN + '/models/model0.pt'],
					P.KEY_PRE_NET_OUTPUTS: ['bn' + str(l)],
				}

				hebb_fc_on_hebb_layer_ft[str(l) + '_' + lrn_rule + '_' + ds + da_names[da]] = {
					P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
					P.KEY_NET_MODULES: 'models.hebb.fc.Net',
					P.KEY_NET_OUTPUTS: 'fc',
					P.KEY_DATA_MANAGER: data_managers[ds],
					P.KEY_AUGMENT_MANAGER: da_managers[da],
					P.KEY_AUGM_BEFORE_STATS: True,
					P.KEY_AUGM_STAT_PASSES: da_mult[da],
					P.KEY_WHITEN: None if lrn_rule_keys[lrn_rule] != 'hwta' else 2,
					P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
					P.KEY_BATCHSIZE: batch_sizes[ds],
					P.KEY_INPUT_SHAPE: input_shapes[ds],
					P.KEY_NUM_EPOCHS: 20,
					P.KEY_OPTIM_MANAGER: 'neurolab.optimization.optim.SGDOptimManager',
					P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
					P.KEY_TOPKACC_K: [1, 5],
					P.KEY_LEARNING_RATE: 1e-3,
					P.KEY_LOCAL_LRN_RULE: lrn_rule_keys[lrn_rule],
					PP.KEY_COMPETITIVE_ACT: lrn_rule_competitive_act[lrn_rule],
					PP.KEY_COMPETITIVE_K: lrn_rule_k[lrn_rule],
					P.KEY_PRE_NET_MODULES: ['models.hebb.model_' + str(num_layers[ds]) + 'l.Net'],
					P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/hebb/gdes_fc_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + da_names[da] + ']/iter' + P.STR_TOKEN + '/models/model0.pt'],
					P.KEY_PRE_NET_OUTPUTS: ['bn' + str(l)],
				}

				gdes_fc2_on_hebb_layer[str(l) + '_' + lrn_rule + '_' + ds + da_names[da]] = {
					P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
					P.KEY_NET_MODULES: 'models.gdes.fc2.Net', 'mdl1:' + PP.KEY_NUM_HIDDEN: 256,
					P.KEY_NET_OUTPUTS: 'fc2',
					P.KEY_DATA_MANAGER: data_managers[ds],
					P.KEY_AUGMENT_MANAGER: da_managers[da],
					P.KEY_AUGM_BEFORE_STATS: True,
					P.KEY_AUGM_STAT_PASSES: da_mult[da],
					P.KEY_WHITEN: None if lrn_rule_keys[lrn_rule] != 'hwta' else 2,
					P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
					P.KEY_BATCHSIZE: batch_sizes[ds],
					P.KEY_INPUT_SHAPE: input_shapes[ds],
					P.KEY_NUM_EPOCHS: da_mult[da] * 20,
					P.KEY_OPTIM_MANAGER: 'neurolab.optimization.optim.SGDOptimManager',
					P.KEY_SCHED_MANAGER: 'neurolab.optimization.sched.MultiStepSchedManager',
					P.KEY_LOSS_METRIC_MANAGER: 'neurolab.optimization.metric.CrossEntMetricManager',
					P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
					P.KEY_TOPKACC_K: [1, 5],
					P.KEY_LEARNING_RATE: 1e-3,
					P.KEY_LR_DECAY: 0.5 if da == 'no_da' else 0.1,
					P.KEY_MILESTONES: range(10, 20) if da == 'no_da' else [20, 30] if da == 'light_da' else [40, 70, 90],
					P.KEY_MOMENTUM: 0.9,
					P.KEY_L2_PENALTY: 5e-4,
					P.KEY_DROPOUT_P: 0.5,
					P.KEY_LOCAL_LRN_RULE: lrn_rule_keys[lrn_rule],
					PP.KEY_COMPETITIVE_ACT: lrn_rule_competitive_act[lrn_rule],
					PP.KEY_COMPETITIVE_K: lrn_rule_k[lrn_rule],
					P.KEY_PRE_NET_MODULES: ['models.hebb.model_' + str(num_layers[ds]) + 'l.Net'],
					P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/hebb/config_base_hebb[' + lrn_rule + '_' + ds + da_names[da] + ']/iter' + P.STR_TOKEN + '/models/model0.pt'],
					P.KEY_PRE_NET_OUTPUTS: ['bn' + str(l)],
				}

				gdes_fc2_on_hebb_layer_ft[str(l) + '_' + lrn_rule + '_' + ds + da_names[da]] = {
					P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
					P.KEY_NET_MODULES: ['models.hebb.model_' + str(num_layers[ds]) + 'l.Net', 'models.gdes.fc2.Net'], 'mdl1:' + PP.KEY_NUM_HIDDEN: 256,
					P.KEY_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/hebb/config_base_hebb[' + lrn_rule + '_' + ds + da_names[da] + ']/iter' + P.STR_TOKEN + '/models/model0.pt'],
					P.KEY_NET_OUTPUTS: ['bn' + str(l), 'fc2'],
					P.KEY_DATA_MANAGER: data_managers[ds],
					P.KEY_AUGMENT_MANAGER: da_managers[da],
					P.KEY_AUGM_BEFORE_STATS: True,
					P.KEY_AUGM_STAT_PASSES: da_mult[da],
					P.KEY_WHITEN: None if lrn_rule_keys[lrn_rule] != 'hwta' else 2,
					P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
					P.KEY_BATCHSIZE: batch_sizes[ds],
					P.KEY_INPUT_SHAPE: input_shapes[ds],
					P.KEY_NUM_EPOCHS: da_mult[da] * 20,
					P.KEY_OPTIM_MANAGER: 'neurolab.optimization.optim.SGDOptimManager',
					P.KEY_SCHED_MANAGER: 'neurolab.optimization.sched.MultiStepSchedManager',
					P.KEY_LOSS_METRIC_MANAGER: 'neurolab.optimization.metric.CrossEntMetricManager',
					P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
					P.KEY_TOPKACC_K: [1, 5],
					P.KEY_LEARNING_RATE: 1e-3,
					P.KEY_ALPHA_L: 0, P.KEY_ALPHA_G: 1,
					P.KEY_LR_DECAY: 0.5 if da == 'no_da' else 0.1,
					P.KEY_MILESTONES: range(10, 20) if da == 'no_da' else [20, 30] if da == 'light_da' else [40, 70, 90],
					P.KEY_MOMENTUM: 0.9,
					P.KEY_L2_PENALTY: l2_penalties[ds + da_names[da]],
					P.KEY_DROPOUT_P: 0.5,
					P.KEY_LOCAL_LRN_RULE: lrn_rule_keys[lrn_rule],
					PP.KEY_COMPETITIVE_ACT: lrn_rule_competitive_act[lrn_rule],
					PP.KEY_COMPETITIVE_K: lrn_rule_k[lrn_rule],
					P.KEY_DEEP_TEACHER_SIGNAL: lrn_rule_dts[lrn_rule],
				}

				hebb_fc2_on_hebb_layer[str(l) + '_' + lrn_rule + '_' + ds + da_names[da]] = {
					P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
					P.KEY_NET_MODULES: 'models.hebb.fc2.Net', 'mdl1:' + PP.KEY_NUM_HIDDEN: 256,
					P.KEY_NET_OUTPUTS: 'fc2',
					P.KEY_DATA_MANAGER: data_managers[ds],
					P.KEY_AUGMENT_MANAGER: da_managers[da],
					P.KEY_AUGM_BEFORE_STATS: True,
					P.KEY_AUGM_STAT_PASSES: da_mult[da],
					P.KEY_WHITEN: None if lrn_rule_keys[lrn_rule] != 'hwta' else 2,
					P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
					P.KEY_BATCHSIZE: batch_sizes[ds],
					P.KEY_INPUT_SHAPE: input_shapes[ds],
					P.KEY_NUM_EPOCHS: 20,
					P.KEY_OPTIM_MANAGER: 'neurolab.optimization.optim.SGDOptimManager',
					P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
					P.KEY_TOPKACC_K: [1, 5],
					P.KEY_LEARNING_RATE: 1e-3,
					P.KEY_LOCAL_LRN_RULE: lrn_rule_keys[lrn_rule],
					PP.KEY_COMPETITIVE_ACT: lrn_rule_competitive_act[lrn_rule],
					PP.KEY_COMPETITIVE_K: lrn_rule_k[lrn_rule],
					P.KEY_DEEP_TEACHER_SIGNAL: lrn_rule_dts[lrn_rule],
					P.KEY_PRE_NET_MODULES: ['models.hebb.model_' + str(num_layers[ds]) + 'l.Net'],
					P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/hebb/config_base_hebb[' + lrn_rule + '_' + ds + da_names[da] + ']/iter' + P.STR_TOKEN + '/models/model0.pt'],
					P.KEY_PRE_NET_OUTPUTS: ['bn' + str(l)],
				}

				hebb_fc2_on_hebb_layer_ft[str(l) + '_' + lrn_rule + '_' + ds + da_names[da]] = {
					P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
					P.KEY_NET_MODULES: 'models.hebb.fc2.Net', 'mdl1:' + PP.KEY_NUM_HIDDEN: 256,
					P.KEY_NET_OUTPUTS: 'fc2',
					P.KEY_DATA_MANAGER: data_managers[ds],
					P.KEY_AUGMENT_MANAGER: da_managers[da],
					P.KEY_AUGM_BEFORE_STATS: True,
					P.KEY_AUGM_STAT_PASSES: da_mult[da],
					P.KEY_WHITEN: None if lrn_rule_keys[lrn_rule] != 'hwta' else 2,
					P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
					P.KEY_BATCHSIZE: batch_sizes[ds],
					P.KEY_INPUT_SHAPE: input_shapes[ds],
					P.KEY_NUM_EPOCHS: 20,
					P.KEY_OPTIM_MANAGER: 'neurolab.optimization.optim.SGDOptimManager',
					P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
					P.KEY_TOPKACC_K: [1, 5],
					P.KEY_LEARNING_RATE: 1e-3,
					P.KEY_LOCAL_LRN_RULE: lrn_rule_keys[lrn_rule],
					PP.KEY_COMPETITIVE_ACT: lrn_rule_competitive_act[lrn_rule],
					PP.KEY_COMPETITIVE_K: lrn_rule_k[lrn_rule],
					P.KEY_DEEP_TEACHER_SIGNAL: lrn_rule_dts[lrn_rule],
					P.KEY_PRE_NET_MODULES: ['models.hebb.model_' + str(num_layers[ds]) + 'l.Net'],
					P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/hebb/gdes_fc_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + da_names[da] + ']/iter' + P.STR_TOKEN + '/models/model0.pt'],
					P.KEY_PRE_NET_OUTPUTS: ['bn' + str(l)],
				}

				prec_on_hebb_layer[str(l) + '_' + lrn_rule + '_' + ds + da_names[da]] = {
					P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
					P.KEY_NET_MODULES: 'neurolab.model.skclassif.Retriever',
					P.KEY_NET_OUTPUTS: 'clf',
					P.KEY_DATA_MANAGER: data_managers[ds],
					P.KEY_AUGMENT_MANAGER: None,
					P.KEY_AUGM_BEFORE_STATS: True,
					P.KEY_AUGM_STAT_PASSES: da_mult[da],
					P.KEY_WHITEN: None if lrn_rule_keys[lrn_rule] != 'hwta' else 2,
					P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
					P.KEY_NUM_TRN_SAMPLES: retr_num_samples[ds],
					P.KEY_BATCHSIZE: batch_sizes[ds],
					P.KEY_INPUT_SHAPE: input_shapes[ds],
					P.KEY_NUM_EPOCHS: 1,
					P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.PrecMetricManager', 'neurolab.optimization.metric.MAPMetricManager'] * len(retr_k[ds]),
					P.KEY_SKCLF_NUM_SAMPLES: retr_num_samples[ds],
					P.KEY_NYSTROEM_N_COMPONENTS: retr_num_nyst[ds],
					P.KEY_KNN_N_NEIGHBORS: retr_num_rel[ds],
					P.KEY_RETR_NUM_REL: retr_num_rel[ds],
					P.KEY_RETR_K: retr_k[ds],
					P.KEY_LOCAL_LRN_RULE: lrn_rule_keys[lrn_rule],
					PP.KEY_COMPETITIVE_ACT: lrn_rule_competitive_act[lrn_rule],
					PP.KEY_COMPETITIVE_K: lrn_rule_k[lrn_rule],
					P.KEY_PRE_NET_MODULES: ['models.hebb.model_' + str(num_layers[ds]) + 'l.Net', 'models.gdes.fc2.Net'], 'mdl1:' + PP.KEY_NUM_HIDDEN: 256,
					P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/hebb/config_base_hebb['+ lrn_rule + '_' + ds + da_names[da] + ']/iter' + P.STR_TOKEN + '/models/model0.pt',
											  P.PROJECT_ROOT + '/results/configs/vision/hebb/gdes_fc2_on_hebb_layer[' + str(l) + '_' + lrn_rule + '_' + ds + da_names[da] + ']/iter' + P.STR_TOKEN + '/models/model0.pt'],
					P.KEY_PRE_NET_OUTPUTS: ['bn' + str(l), 'bn1'],
				}

				prec_on_hebb_layer_ft[str(l) + '_' + lrn_rule + '_' + ds + da_names[da]] = {
					P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
					P.KEY_NET_MODULES: 'neurolab.model.skclassif.Retriever',
					P.KEY_NET_OUTPUTS: 'clf',
					P.KEY_DATA_MANAGER: data_managers[ds],
					P.KEY_AUGMENT_MANAGER: None,
					P.KEY_AUGM_BEFORE_STATS: True,
					P.KEY_AUGM_STAT_PASSES: da_mult[da],
					P.KEY_WHITEN: None if lrn_rule_keys[lrn_rule] != 'hwta' else 2,
					P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
					P.KEY_NUM_TRN_SAMPLES: retr_num_samples[ds],
					P.KEY_BATCHSIZE: batch_sizes[ds],
					P.KEY_INPUT_SHAPE: input_shapes[ds],
					P.KEY_NUM_EPOCHS: 1,
					P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.PrecMetricManager', 'neurolab.optimization.metric.MAPMetricManager'] * len(retr_k[ds]),
					P.KEY_SKCLF_NUM_SAMPLES: retr_num_samples[ds],
					P.KEY_NYSTROEM_N_COMPONENTS: retr_num_nyst[ds],
					P.KEY_KNN_N_NEIGHBORS: retr_num_rel[ds],
					P.KEY_RETR_NUM_REL: retr_num_rel[ds],
					P.KEY_RETR_K: retr_k[ds],
					P.KEY_LOCAL_LRN_RULE: lrn_rule_keys[lrn_rule],
					PP.KEY_COMPETITIVE_ACT: lrn_rule_competitive_act[lrn_rule],
					PP.KEY_COMPETITIVE_K: lrn_rule_k[lrn_rule],
					P.KEY_PRE_NET_MODULES: ['models.hebb.model_' + str(num_layers[ds]) + 'l.Net', 'models.gdes.fc2.Net'], 'mdl1:' + PP.KEY_NUM_HIDDEN: 256,
					P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/hebb/gdes_fc2_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + da_names[da] + ']/iter' + P.STR_TOKEN + '/models/model0.pt',
											  P.PROJECT_ROOT + '/results/configs/vision/hebb/gdes_fc2_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + da_names[da] + ']/iter' + P.STR_TOKEN + '/models/model1.pt'],
					P.KEY_PRE_NET_OUTPUTS: ['bn' + str(l), 'bn1'],
				}

				knn_on_hebb_layer[str(l) + '_' + lrn_rule + '_' + ds + da_names[da]] = {
					P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
					P.KEY_NET_MODULES: 'neurolab.model.skclassifKNNClassifier',
					P.KEY_NET_OUTPUTS: 'clf',
					P.KEY_DATA_MANAGER: data_managers[ds],
					P.KEY_AUGMENT_MANAGER: da_managers[da],
					P.KEY_AUGM_BEFORE_STATS: True,
					P.KEY_AUGM_STAT_PASSES: da_mult[da],
					P.KEY_WHITEN: None if lrn_rule_keys[lrn_rule] != 'hwta' else 2,
					P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
					P.KEY_BATCHSIZE: batch_sizes[ds],
					P.KEY_INPUT_SHAPE: input_shapes[ds],
					P.KEY_NUM_EPOCHS: da_mult[da],
					P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
					P.KEY_TOPKACC_K: [1, 5],
					P.KEY_SKCLF_NUM_SAMPLES: da_mult[da] * tot_trn_samples[ds],
					P.KEY_NYSTROEM_N_COMPONENTS: 1000,
					P.KEY_KNN_N_NEIGHBORS: 10,
					P.KEY_LOCAL_LRN_RULE: lrn_rule_keys[lrn_rule],
					PP.KEY_COMPETITIVE_ACT: lrn_rule_competitive_act[lrn_rule],
					PP.KEY_COMPETITIVE_K: lrn_rule_k[lrn_rule],
					P.KEY_PRE_NET_MODULES: ['models.hebb.model_' + str(num_layers[ds]) + 'l.Net'],
					P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/hebb/config_base_hebb[' + lrn_rule + '_' + ds + da_names[da] + ']/iter' + P.STR_TOKEN + '/models/model0.pt'],
					P.KEY_PRE_NET_OUTPUTS: ['bn' + str(l)],
				}

				knn_on_hebb_layer_ft[str(l) + '_' + lrn_rule + '_' + ds + da_names[da]] = {
					P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
					P.KEY_NET_MODULES: 'neurolab.model.skclassifKNNClassifier',
					P.KEY_NET_OUTPUTS: 'clf',
					P.KEY_DATA_MANAGER: data_managers[ds],
					P.KEY_AUGMENT_MANAGER: da_managers[da],
					P.KEY_AUGM_BEFORE_STATS: True,
					P.KEY_AUGM_STAT_PASSES: da_mult[da],
					P.KEY_WHITEN: None if lrn_rule_keys[lrn_rule] != 'hwta' else 2,
					P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
					P.KEY_BATCHSIZE: batch_sizes[ds],
					P.KEY_INPUT_SHAPE: input_shapes[ds],
					P.KEY_NUM_EPOCHS: da_mult[da],
					P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
					P.KEY_TOPKACC_K: [1, 5],
					P.KEY_SKCLF_NUM_SAMPLES: da_mult[da] * tot_trn_samples[ds],
					P.KEY_NYSTROEM_N_COMPONENTS: 1000,
					P.KEY_KNN_N_NEIGHBORS: 10,
					P.KEY_LOCAL_LRN_RULE: lrn_rule_keys[lrn_rule],
					PP.KEY_COMPETITIVE_ACT: lrn_rule_competitive_act[lrn_rule],
					PP.KEY_COMPETITIVE_K: lrn_rule_k[lrn_rule],
					P.KEY_PRE_NET_MODULES: ['models.hebb.model_' + str(num_layers[ds]) + 'l.Net'],
					P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/hebb/gdes_fc_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + da_names[da] + ']/iter' + P.STR_TOKEN + '/models/model0.pt'],
					P.KEY_PRE_NET_OUTPUTS: ['bn' + str(l)],
				}

				svm_on_hebb_layer[str(l) + '_' + lrn_rule + '_' + ds + da_names[da]] = {
					P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
					P.KEY_NET_MODULES: 'neurolab.model.skclassif.SVMClassifier',
					P.KEY_NET_OUTPUTS: 'clf',
					P.KEY_DATA_MANAGER: data_managers[ds],
					P.KEY_AUGMENT_MANAGER: da_managers[da],
					P.KEY_AUGM_BEFORE_STATS: True,
					P.KEY_AUGM_STAT_PASSES: da_mult[da],
					P.KEY_WHITEN: None if lrn_rule_keys[lrn_rule] != 'hwta' else 2,
					P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
					P.KEY_BATCHSIZE: batch_sizes[ds],
					P.KEY_INPUT_SHAPE: input_shapes[ds],
					P.KEY_NUM_EPOCHS: da_mult[da],
					P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
					P.KEY_TOPKACC_K: [1, 5],
					P.KEY_SKCLF_NUM_SAMPLES: da_mult[da] * tot_trn_samples[ds],
					P.KEY_NYSTROEM_N_COMPONENTS: 1000,
					P.KEY_LOCAL_LRN_RULE: lrn_rule_keys[lrn_rule],
					PP.KEY_COMPETITIVE_ACT: lrn_rule_competitive_act[lrn_rule],
					PP.KEY_COMPETITIVE_K: lrn_rule_k[lrn_rule],
					P.KEY_PRE_NET_MODULES: ['models.hebb.model_' + str(num_layers[ds]) + 'l.Net'],
					P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/hebb/config_base_hebb[' + lrn_rule + '_' + ds + da_names[da] + ']/iter' + P.STR_TOKEN + '/models/model0.pt'],
					P.KEY_PRE_NET_OUTPUTS: ['bn' + str(l)],
				}

				svm_on_hebb_layer_ft[str(l) + '_' + lrn_rule + '_' + ds + da_names[da]] = {
					P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
					P.KEY_NET_MODULES: 'neurolab.model.skclassif.SVMClassifier',
					P.KEY_NET_OUTPUTS: 'clf',
					P.KEY_DATA_MANAGER: data_managers[ds],
					P.KEY_AUGMENT_MANAGER: da_managers[da],
					P.KEY_AUGM_BEFORE_STATS: True,
					P.KEY_AUGM_STAT_PASSES: da_mult[da],
					P.KEY_WHITEN: None if lrn_rule_keys[lrn_rule] != 'hwta' else 2,
					P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
					P.KEY_BATCHSIZE: batch_sizes[ds],
					P.KEY_INPUT_SHAPE: input_shapes[ds],
					P.KEY_NUM_EPOCHS: da_mult[da],
					P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
					P.KEY_TOPKACC_K: [1, 5],
					P.KEY_SKCLF_NUM_SAMPLES: da_mult[da] * tot_trn_samples[ds],
					P.KEY_NYSTROEM_N_COMPONENTS: 1000,
					P.KEY_LOCAL_LRN_RULE: lrn_rule_keys[lrn_rule],
					PP.KEY_COMPETITIVE_ACT: lrn_rule_competitive_act[lrn_rule],
					PP.KEY_COMPETITIVE_K: lrn_rule_k[lrn_rule],
					P.KEY_PRE_NET_MODULES: ['models.hebb.model_' + str(num_layers[ds]) + 'l.Net'],
					P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/hebb/gdes_fc_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + da_names[da] + ']/iter' + P.STR_TOKEN + '/models/model0.pt'],
					P.KEY_PRE_NET_OUTPUTS: ['bn' + str(l)],
				}
