import math 
import numpy as np 
import matplotlib.pyplot as plt
from rnn_models import GRU,RNN

# hyperparamter
hidden_dim = 100
input_dim = 1
output_dim = 1
f = 30
seq_len = 50
epochs = 10
learning_rate = 1e-4
min_val = -200
max_val = 200

# make dataset
seq_data = np.array([(0.5*math.sin(1*f*2*np.pi*x) + 0.5*math.sin(2*f*2*np.pi*x)) for x in np.arange(0,0.4,0.002)]) 
print(len(seq_data))
# visualize the seq_wave
fig = plt.figure(figsize = (12,5))
plt.plot(seq_data)
plt.title('sequence data')


# training dataset
X = []
Y = []
num_records = len(seq_data) - seq_len

print(f'seq_len: {seq_len}')
print(f'num_records: {num_records}')

for i in range(num_records):
    X.append(seq_data[i:i+seq_len])
    Y.append(seq_data[i+seq_len])

X = np.array(X)
Y = np.array(Y)
X = np.expand_dims(X,2) # reshape to (training_sample,seq_len,1)
Y = np.expand_dims(Y,1) # (training_sample,1)

print(f'X.shape: {X.shape}')
print(f'Y.shape: {Y.shape}')

# testing dataset
X_val = []
Y_val = []
seq_len = 50
num_records = len(seq_data) - seq_len

for i in range(num_records-seq_len,num_records):
    X_val.append(seq_data[i:i+seq_len])
    Y_val.append(seq_data[i+seq_len])

X_val = np.array(X_val)
Y_val = np.array(Y_val)
X_val = np.expand_dims(X_val,2) # reshape to (training_sample,seq_len,1)
Y_val = np.expand_dims(Y_val,1) # (training_sample,1)

print(f'X_val.shape: {X_val.shape}')
print(f'Y_val.shape: {Y_val.shape}')

if __name__ == "__main__":
	node = GRU()

	# train
	losses = node.train(X,Y,epochs=epochs,learning_rate=1e-4,min_val = min_val,max_val = max_val,predict = False)

	# predict
	preds = node.predict(X)
	plt.plot(preds,label = 'pred')
	plt.plot(Y,label = 'ground-truth')
	plt.title('fater training')
	plt.legend()

	#plt.savefig('predict.png')
	plt.figure()
	plt.plot(losses,label = 'losses')
	plt.title('loss-curve')

	plt.show()