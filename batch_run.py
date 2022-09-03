import os

# test
os.system("python test.py --content_path /home/xianglin/projects/DVI_data/resnet18_mnist -d mnist_full -g 3")
os.system("python test.py --content_path /home/xianglin/projects/DVI_data/resnet18_fmnist -d fmnist_full -g 3")
os.system("python test.py --content_path /home/xianglin/projects/DVI_data/resnet18_cifar10 -d cifar10_full -g 3")
