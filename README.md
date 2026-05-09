# -
期中作业
环境：
Python 3.13
PyTorch 2.11.0+cu128
TorchVision 0.26.0+cu128
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
数据准备：python prepare_pet_data.py 解压数据至data/oxford-iiit-pet/
运行试验：python run_task1_experiments.py --epochs 10 --batch-size 32 --num-workers 2 --device cuda --output-dir runs/task1_submit
汇总实验结果：python collect_task1_results.py --runs-dir runs/task1_submit
生成所有实验验证集 Accuracy 曲线：python plot_task1_metrics.py runs/task1_submit/baseline_resnet18_imagenet_lr5e-5_5e-4
训练完成后，每个实验目录下会生成：
config.json
metrics.csv
summary.json
best.pt
last.pt
learning_curves.png
