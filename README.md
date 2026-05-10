# -
期中作业
环境：
Python 3.13
PyTorch 2.11.0+cu128
TorchVision 0.26.0+cu128
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
数据准备：prepare_pet_data.py 
运行试验：run_task1_experiments.py -
汇总实验结果：python collect_task1_results.py 
生成所有实验验证集 Accuracy 曲线：python plot_task1_metrics.py 
训练完成后，每个实验目录下会生成：
config.json
metrics.csv
summary.json
best.pt
last.pt
learning_curves.png
