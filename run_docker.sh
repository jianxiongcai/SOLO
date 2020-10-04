# Networking:
# bind port: host (8890) => jupyter lab (8888)
# bind port: host (6008) => tensorboard (6006)

# Directionary:
# /workspace/data: the data folder
# /workspace/code: the code folder

docker run -it -p 8890:8888 -p 6008:6006 --gpus all -v /home/jianxiong/data/HW3:/workspace/data -v /home/jianxiong/Desktop/CIS680-HW3:/workspace/CIS680-HW3 jianxiongcai/robotfly-pytorch
