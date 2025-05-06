import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import keras._tf_keras.keras.utils as utils
import keras._tf_keras.keras.optimizers as optimizers
import keras._tf_keras.keras.losses as losses
from keras._tf_keras.keras.models import Sequential
import keras._tf_keras.keras.layers as layers
from keras._tf_keras.keras.datasets import mnist
import keras._tf_keras.keras.activations as activations
import seaborn as sns

#np.random.seed(0)
def threshold(z): #keep binary image
    return 1 if z > 0 else 0

def performance_plot(nn_fit_history):
    plt.figure(figsize=(16, 6))

    #plot loss
    plt.subplot(1, 2, 1)
    plt.plot(nn_fit_history['loss'])
    plt.plot(nn_fit_history['val_loss'])
    plt.ylabel('loss', size=12)
    plt.xlabel('epoch', size=12)
    plt.legend(['train', 'val'])

    #plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(nn_fit_history['accuracy'])
    plt.plot(nn_fit_history['val_accuracy'])
    plt.ylabel('accuracy', size=12)
    plt.xlabel('epoch', size=12)
    plt.legend(['train', 'val'])

    plt.show()

MNN = lambda input_shape:Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(), #the 2d array becomes 1d
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax") #output 10 possibile numbers [0:9]
    ])

'''
dropout layer helps prevent overfitting between interconnected layers nodes
Dropout Rate (p): This is a hyperparameter that specifies the probability of a neuron being dropped out.
Common values for p range from 0.2 to 0.5. A rate of 0.5 means that, on average, half of the neurons 
will be deactivated during each training step.
It reduces Interdependencies Between Neurons
'''

CNN = lambda input_shape:Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape= input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(), #startin the fully connected part of NN #the 2d array becomes 1d
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax") #output 10 possibile numbers [0:9]
    ])

if __name__ == "__main__":
    x = np.linspace(-4, 4, 1000) 
    plt.figure(figsize=(16, 10))
    plt.title("Various activation functions")
    plt.plot(x, list(map(threshold, x)), label="threshold")
    plt.plot(x, activations.relu(x), label="ReLU")
    plt.plot(x, activations.leaky_relu(x, negative_slope=0.1), dashes=(3, 3), linewidth=3, label="leaky ReLU")
    plt.plot(x, activations.sigmoid(x), label="sigmoid")
    plt.plot(x, activations.tanh(x), label="tanh")
    plt.legend()
    plt.show()

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # #rescaling RGB to binary scale 0/1
    X_train = X_train/255
    X_test = X_test/255

    print(f'shape {X_train.shape}\n', X_train) #2D dataset

    # plt.figure()
    # plt.scatter(X_train[:,0], X_train[:,1], c=y_train, edgecolors='k', s=20) #for simplicity of visualiztion, we'll consider the first column as x, and the second column as y 
    # plt.show()

    plt.figure(figsize=(8, 8))
    grid_shape = (4, 5)
    for digit_num in range(grid_shape[0] * grid_shape[1]):
        plt.subplot(*grid_shape,digit_num+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(X_train[digit_num], interpolation = "none", cmap = "bone_r")
        plt.xlabel(f'label: {y_train[digit_num]}', size=12)
    #plt.show()

    X_train = np.expand_dims(X_train, axis=-1) #for CNN
    print(f'shape: {X_train[0].shape}')
    #model = MNN(X_train[0].shape)
    model = CNN(X_train[0].shape)
    print(model.summary())

    # utils.plot_model(
    #     model, 
    #     show_shapes=True,
    #     show_dtype= True,
    #     show_layer_names=True,
    #     rankdir='TB',
    #     expand_nested=False,
    #     dpi=96,
    #     range = None,
    #     show_layer_activations=True
    # )

    #SparseCategoricalCrossentropy is used for multiclass classification
    model.compile(
        optimizer= optimizers.Adam(),
        loss= losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    '''
    The training dataset is divided into smaller batches of size BATCH_SIZE.
    
    For each batch:
    -The network performs a forward pass, calculating the predictions for all examples in the batch.
    -The loss function is computed based on the predictions and the true labels for this batch.
    -The gradients of the loss with respect to the network's weights are calculated (using backpropagation) based on this batch.   
    -The optimization algorithm (e.g., Adam, SGD) uses these gradients to update the network's weights.
    
    An epoch represents one complete pass through the entire training dataset. During one epoch, the network sees every training example once.
    '''
    BATCH_SIZE = 120
    EPOCHS = 5
    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test)
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    
    '''
    The Validation Loss Starts to Increase While the Training Loss Continues to Decrease (or Plateaus at a Low Value).
    
    What to look for in the plot:
    -Training Loss: Should generally decrease over epochs.
    -Validation Loss: Should initially decrease but will start to increase or plateau when the model begins to overfit
    '''
    performance_plot(history.history) #we'd need to use regularization in case of overfitting

    #predict_image = np.expand_dims(X_test[0], 0)
    prediction = model.predict(np.expand_dims(X_test[0], 0)) #predict method expect list of images, meanwhile we need to predict just one
    print(f'{np.argmax(prediction)}, true label: {y_test[0]}')

    


