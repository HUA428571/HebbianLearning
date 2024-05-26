configs.vision.gdes.config_base_gdes[cifar10]
configs.vision.gdes.gdes_fc_on_gdes_layer[5_cifar10]
configs.vision.hebb.config_base_hebb[1wta_cifar10]
configs.vision.hebb.gdes_fc_on_hebb_layer_ft[1_1wta_cifar10]

nohup python runstack.py --stack stacks.vision.gdes[cifar10] --mode train --device cuda:1 --clearhist > gdes.out

nohup python runstack.py --stack stacks.vision.hebb[eswta_mnist] --mode train --device cuda:3 >> hebb_eswta_mnist.out

nohup python runstack.py --stack stacks.vision.hebb[5wta_mnist] --mode train --device cuda:3 > hebb_5wta_mnist.out
nohup python runstack.py --stack stacks.vision.hebb[5wta_cifar10] --mode train --device cuda:3 > hebb_5wta.out

nohup python runstack.py --stack stacks.vision.hebb[pswta_mnist] --mode train --device cuda:2 >> hebb_pswta_mnist.out

nohup python runstack.py --stack stacks.vision.hebb[1wta_mnist] --mode train --device cuda:2 > hebb_1wta_mnist.out

