'''
Base on: https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py
'''

import numpy as np   
import matplotlib.pyplot as plt  
import itertools  
import os
import zipfile
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
import datetime
import tensorflow as tf

# Create a function to import an image and resize it to be able to be used with your model
def load_and_prep_image(filename,image_shape = 224, scale = True):
    '''
    Reload in an image form filename, turn it into a tensor and reshapes into (224,224,3)
    
    Arguments:
        filename (str) --- string filname of target image
        img_shape (int) --- size to resize target image to, default 224
        scale (bool) --- whether to scale pixel values to range(0,1) default: True
    
    Return:
        img (tensor) --- image processed
    
    Exmaple:
        img = load_and_prep_image("./Pound_layer_cake.jpg",
                       image_shape = 224,
                       scale = True)
    '''
    # Read the image
    img = tf.io.read_file(filename)
    # Decode it into tensor
    img_tensor = tf.image.decode_jpeg(img)
    # Resize the image
    img_tensor = tf.image.resize(img_tensor,[image_shape,image_shape])
    if scale:
        # Rescale the image (Normalization)
        return img_tensor/255.0
    else:
        return img_tensor

# plot confusion matrix, base on scikit-learn
def plot_confusion_matrix(y_true,y_pred,classes= None,figsize = (10,10),
                         text_size = 15,norm = False,savefig = True):
    '''
    Makes a labelled confusion matrix comparing predictions and ground truth labels
    If classes is passed, confusion matrix will be labelled, if not, integer class values will be used
    
    Arguments:
        y_true (Array) --- Array of truth labels (must ne same shape as y_pred)
        y_pred (Array) --- Array of predicted labels (must be same shape as y_true)
        classes (Array) --- Array of classes labels (e.g string form), If 'None' integer labels are used
        figsize (Tuple) --- Size of output figure text (default = (10,10))
        text_size --- Size of ouput figure text (default = 15)
        norm (Bool) --- Normalize values or not (default = False)
        savefig (Bool) --- Save your confusion matrix figure or not (default = True)
    Return:
        Show a labelled confusion matrix plot comaring y_true and y_pred
    
    Example:
        plot_confusion_matrix(y_true = test_labels, # ground truth test labels
                                y_pred = y_preds, # predicted labels
                                classes = class_name. # array of class label names
                                figsize = (15,15),
                                text_size = 10
                                
                                )
    '''
    # Create confusion matrix
    cm = confusion_matrix(y_true,y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis = 1)[:,np.newaxis] # normalize it
    n_classes = cm.shape[0] # find the number of classes we are dealing with
    
    # Plot the figure and make it pretty
    fig,ax = plt.subplots(figsize = figsize)
    cax = ax.matshow(cm,cmap = plt.cm.Blues) # colors will represent how correct a class is, darker == better
    fig.colorbar(cax)
    
    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])
        
    # Label the axes
    ax.set(title = 'Confusion Matrix',
           xlabel = "Predicted label",
           ylabel = "True label",
           xticks = np.arange(n_classes),
           yticks = np.arange(n_classes),
           xticklabels = labels, # axes will labeled with class name(if they exist) or ints
           yticklabels = labels
          )
    
    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()
    
    # Set the threshold for different color
    threshold = (cm.max() + cm.min())/2
    
    # Plot the text on each cell
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        if norm:
            plt.text(j,i,f"{cm[i,j]} ({cm_norm[i,j]*100:.1f}%)",
                     horizontalalignment = 'center',
                     color = 'white' if cm[i,j] > threshold else "black",
                     size = text_size
                    )
    # Save the figure to current working directory
    if savefig:
        fig.savefig('confusion_matrix.png')    

# Make function to predict on images and plot them (works with multi-class)
def pred_and_plot(model,filename,class_names):
    '''
    Imports an image located at filename, makes a prediction on it with
    a trained model an plots the image with the predicted class as the title.
    
    Arguments:
        model(machine learning model) --- your model
        filename(string) --- your image's name
        class_names(list) --- your predict labels
    
    Return:
        predicted label and display the image
        
    '''
    # import the target image and preprocess it
    img = load_and_prep_image(filename)
    # make a prediction
    pred = model.predict(tf.expand_dims(img,axis = 0)) # (1,height,width,color channels)
    # get the predicted class
    if len(pred[0]) >1: # check for multi class
        pred_class = class_names[pred.argmax()] # if more than one output take the max
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])] # if binary output, round
    #plot the image and predicted class
    plt.imshow(img)
    plt.title(f'Prediction: {pred_class}')
    plt.axis(False)

# Define a function to create tensorboard callback
def create_tensorboard_callback(dir_name,experiment_name):
    '''
    Create a tensorboard callback instand to store log files,
    Stores log files with filepath:
        "dir_name/experiment_name/current_datetime/"
        
    Arguments:
        dir_name --- target directory to store tensorboard log files
        experiment_name --- name of experiment directory (e.g efficientnet_model_1)
    
    Return:
        tensorboard callback
    '''
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback =  tf.keras.callbacks.TensorBoard(log_dir = log_dir)
    print(f'Saving TensorBoard log files to: {log_dir}')
    return tensorboard_callback

def plot_loss_curves(history):
    '''
    Returns sepatate loss curves for training and validation metrics
    
    Arguments:
        history: TensorFlow model History object
        
    Return:
        Display seperately loss and accuracy of train and validation
    '''
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(len(loss))
    #Plot loss curve
    plt.plot(epochs,loss,label = 'training_loss')
    plt.plot(epochs,val_loss,label = 'validation_loss')
    plt.title('Loss')
    plt.legend()
    # Plot accuracy curve
    plt.figure()
    plt.plot(epochs,accuracy,label = 'training_accuracy')
    plt.plot(epochs,val_accuracy,label = 'validation_accuracy')
    plt.title('Accuracy')
    plt.legend()

def compare_histories(original_history,new_history,initial_epochs= 5):
    '''
    Compares two TensorFlow model history objects
    
    Arguments:
        original_history --- History object of original model (before new history)
        new_history --- History object from continued model training (after original_history)
        initial_epochs --- Number of epochs in original_history (new history plot starts from here)
    '''
    # Get original history measurements
    acc = original_history.history['accuracy']
    loss = original_history.history['loss']
    
    val_acc = original_history.history['val_accuracy']
    val_loss = original_history.history['val_loss']
    
    # Combine original with new history
    total_acc = acc + new_history.history['accuracy']
    total_loss = loss + new_history.history['loss']
    
    total_val_acc = val_acc + new_history.history['val_accuracy']
    total_val_loss = val_loss + new_history.history['val_loss']
    
    print(len(total_acc))
    print(total_acc)
    
    # Make plot
    # Accuracy plot
    plt.figure(figsize = (8,8))
    plt.subplot(2,1,1)
    plt.plot(total_acc,label = 'Training accuracy')
    plt.plot(total_val_acc,label = 'Validation_Accuracy')
    plt.plot([initial_epochs -1 ,initial_epochs - 1],plt.ylim(),label = 'Starting fine tuning') # reshift plot around epochs
    plt.legend(loc = 'lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('epoch')
    
    # Loss plot 
    plt.subplot(2,1,2) # row and column
    plt.plot(total_loss,label = 'Training loss')
    plt.plot(total_val_loss,label = 'Validation loss')
    plt.plot([initial_epochs -1,initial_epochs -1],plt.ylim(),label = 'Starting fine tuning')
    plt.legend(loc = 'upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    
# Create function to unzip and zipfile into currently working directory
# (since we are going to be downloading and unzipping a few files)
def unzip_data(filename):
    '''
    Unzips filename into current working directory
    
    Arguments:
        filename(str)--- filepath to target zip floder to be unzipped
    
    Return:
        None
    '''
    zip_ref = zipfile.ZipFile(filename,'r')
    zip_ref.extractall()
    zip_ref.close()

# function to evaluate: accuracy
def calculate_results(y_true,y_pred):
    '''
    Calculates model accuracy, precision, recall, f1-score a binary classification model
    
    Arguments:
        y_true: true labels in the form of a 1D array
        y_pred: predicted labels in the form of 1D array
        
    Return:
        a dictionary of accuracy, precision, recall,f1_score
    '''
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true,y_pred) * 100
    # Calculate model precsion, recall and f1, score using 'weighted average'
    model_precision,model_recall,model_f1,_ = precision_recall_fscore_support(y_true,y_pred,average = 'weighted')
    model_results = {"accuracy":model_accuracy,
                    "precision":model_precision,
                    "recall":model_recall,
                    "f1": model_f1}
    return model_results

# Walk through an image classification directory and find out how many files (images)
# are in each subdirectory

def walk_through_dir(dir_path):
    '''
    Walk through dir_path returning its contents
    
    Arguments:
        dir_path (str) --- target directory
    
    Return:
        A print out of:
            number of subdirectories in dir_path
            number of images (files) in each subdirectories
            name of each subdirectory
    '''
    for dirpath,dirnames,filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'")
        
def predict_image(model,class_names,image_path,image_shape = 224,scale = False,display = False):
    '''
    Read jpeg image and convert to tensor
    
    Arguments:
        model --- tensorflow machine learning model
        class_names --- classes name of the dataset
        image_path --- path to images
        image_shape --- shape of the images default (224x224)
        scale --- normalize image or not
        display --- display the image and result or not
    Return:
        pred_label --- label predicted
    '''
    # read the image
    img = tf.io.read_file(image_path)
    # convert to tensor
    img_tensor = tf.image.decode_jpeg(img)
    # resize image
    tensor = tf.image.resize(img_tensor,(image_shape,image_shape))
    # normalize tensor if scale == True
    if scale:
        tensor = tensor/255.
    # expand dims for image (1,height,width,color_channels)
    tensor = tf.expand_dims(tensor,axis = 0)
    # predict
    pred_label = class_names[model.predict(tensor).argmax()]
    # show display
    if display:
        plt.imshow(img_tensor)
        plt.title(pred_label)
        return pred_label
    else:
        return pred_label
    
def random_predict_16(dataset_dir,model,class_names,image_shape = 224,scale = False,display = False):
    '''
    Read jpeg image and convert to tensor
    
    Arguments:
        dataset_dir --- directory of dataset
        model --- tensorflow machine learning model
        class_names --- classes name of the dataset
        image_shape --- shape of the images default (224x224)
        scale --- normalize image or not
        display --- display the image and result or not
    Return:
        Display image with label predicted
    '''
    images = []
    pred_labels = []
    for label in class_names:
        images = images + [ label + "/" + i for i in os.listdir(dataset_dir + "/" + label)]
    image_16 = random.choices(images,k = 16)
    plt.figure(figsize = (15,15))
    for i,image in enumerate(image_16):
        plt.subplot(4,4,i+1)      
        # read the image
        img = tf.io.read_file(dataset_dir + "/" +image)
        # convert to tensor
        img_tensor = tf.image.decode_jpeg(img)
        # resize image
        tensor = tf.image.resize(img_tensor,(image_shape,image_shape))
        # normalize tensor if scale == True
        if scale:
            tensor = tensor/255.
        # expand dims for image (1,height,width,color_channels)
        tensor = tf.expand_dims(tensor,axis = 0)
        # predict
        pred_label = class_names[model.predict(tensor).argmax()]
        plt.imshow(img_tensor)
        plt.title(pred_label)
    plt.tight_layout()
