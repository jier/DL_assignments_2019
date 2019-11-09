import os

learning_rate = [2e-3, 2e-4, 2e-5]
max_steps = [1500, 3000, 4500, 6000]
batch_size = [50, 100, 200, 250]
dnn_hidden_units = ['100','100,200,100', '200, 400, 200', '200, 200']
optimizer = ['Adam','Adamax','Adagrad','Adadelta']

for dnn in dnn_hidden_units:
    for lr in learning_rate:
        for step in max_steps:
            for bs in batch_size:
                for optim in optimizer:
                    os.system(f"python train_mlp_pytorch.py --dnn_hidden_units {dnn} --learning_rate {lr} --max_steps {step} --batch_size {bs} --optimizer {optim}")