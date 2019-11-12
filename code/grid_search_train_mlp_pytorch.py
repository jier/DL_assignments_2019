import subprocess
import csv

learning_rate = [2e-2, 2e-3, 2e-4]
max_steps = [1500, 3000]
batch_size = [100, 200, 250]
dnn_hidden_units = ['100, 100', '100,300,100','100,200,400,200,100']
optimizer = ['SGD','Adam','Adamax','Adagrad']

columns = ['Iter','acc', 'dnn_hidden', 'lr','batch_size','optimizer']
data = []
print(f"[DEBUG] Training start...")
for dnn in dnn_hidden_units:
    for lr in learning_rate:
        for step in max_steps:
            for bs in batch_size:
                for optim in optimizer:
                    output = subprocess.check_output(['python', 'train_mlp_pytorch.py',f'--dnn_hidden_units={dnn}',f'--learning_rate={lr}',f'--max_steps={step}',f'--batch_size={bs}',f'--optimizer={optim}'])
                    output = output.decode('utf-8').strip()
                    print(f" Accuracy {output} --dnn_hidden_units {dnn} --learning_rate {lr} --max_steps {step} --batch_size {bs} --optimizer {optim}")
                    dict_data = {'Iter':step,'acc':output, 'dnn_hidden':dnn, 'lr':lr,'batch_size':bs,'optimizer':optim}
                    data.append(dict_data)
                    # os.system(f"python train_mlp_pytorch.py --dnn_hidden_units {dnn} --learning_rate {lr} --max_steps {step} --batch_size {bs} --optimizer {optim}")
print("[DEBUG] Done training")
with open('.test.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=columns)
    writer.writeheader()
    for d in data:
        writer.writerow(d)
print("[DEBUG] Wrote to test.csv done.")