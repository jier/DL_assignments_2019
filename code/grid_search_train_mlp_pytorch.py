import subprocess
import csv

learning_rate = [2e-3, 2e-4, 2e-5]
max_steps = [1500]
batch_size = [150, 300]
dnn_hidden_units = ['300,300,400,700,500,700,400,300,300','500,800,800,500,100' ]
optimizer = ['Adam','Adamax','Adagrad']

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
                    print(output)
                    # print(f"train_loss_avg {rloss/(i + 1)}, test_loss_avg {t_loss}, train_acc {train_accuracy}, best_test_accuracy {output} --dnn_hidden_units {dnn} --learning_rate {lr} --max_steps {step} --batch_size {bs} --optimizer {optim}")
                    dict_data = {'Iter':step,'acc':output, 'dnn_hidden':dnn, 'lr':lr,'batch_size':bs,'optimizer':optim}
                    data.append(dict_data)
                print('------------------------------------------------------------------------------------------------------------------\n')
                    # os.system(f"python train_mlp_pytorch.py --dnn_hidden_units {dnn} --learning_rate {lr} --max_steps {step} --batch_size {bs} --optimizer {optim}")
print("[DEBUG] Done training")
with open('.test_deeper.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=columns)
    writer.writeheader()
    for d in data:
        writer.writerow(d)
print("[DEBUG] Wrote to test.csv done.")