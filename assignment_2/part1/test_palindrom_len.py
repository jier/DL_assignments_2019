from train import train
import argparse
import numpy as np

def gen_data(config):

    LENGTHS = range(5, 45)
    MODEL_TYPES = ['RNN', 'LSTM']
    accuracies = []

    for 'RNN' in MODEL_TYPES:
        for l in LENGTHS:
            accuracy_temp = []
            config.input_length = l
            for iter in range(l):
                np.random.seed(iter *l)
                accuracy = train(config)
                accuracy_temp.append(accuracy)
            accuracies.append((np.array(accuracy_temp).mean(),np.array(accuracy_temp).std()))



if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence') # 1
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps') #10000
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")
    # Debug material
    parser.add_argument('--csv', type=str, default='loss_accuracy.csv')
    parser.add_argument('--summary', type=str, default='runs/RNN', help='Specify where to write out tensorboard summaries')
    parser.add_argument('--tensorboard', type=int, default=0, help='Use tensorboard for one run, default do not show')
    parser.add_argument('--record_plot', type=int, default=0, help='Useful when training to save csv data to plot')
    config = parser.parse_args()

    # Train the model
    gen_data(config)
