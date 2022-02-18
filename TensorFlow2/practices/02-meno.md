`from sklearn.datasets import make_circles`

`X,y = make_circles(n_samples,noise = 0.03, random_state = 42)`

`circles = pd.DataFrame({"X0":X[:,0],"X1":X[:,1],"label":y})`

`plt.scatter(X[:,0],X[:,1],c=y, cmap =plt.cm.RdYlBu)`

    
    history_8 = model_8.fit(X_train,
                       y_train,
                       epochs = 100,
                       callbacks = [lr_scheduler],
                       verbose = 1)
                   
`tf.keras.layers.Dense(1,activation = tf.keras.activations.linear`

    
    model_4.compile(loss = "binary_crossentropy",
               optimizer = tf.keras.optmizers.Adam(learning_rate = 0.001),
               metrics= ['accuracy'])
               

`lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch:1e-4 * 10**(epoch/20))`

    A = tf.cast(tf.range(-10,10),tf.float32)
    plt.plot(A)

`history_8 = model_8.fit(X_train,
                       y_train,
                       epochs = 100,
                       callbacks = [lr_scheduler],
                       verbose = 1)`

`plt.semilogx(lrs,history_8.history['loss'])`

    loss, accuracy = model_9.evaluate(X_test,y_test)
    print(f'Model loss on  the test set: {loss}')
    print(f'Model accuracy on the test set: {(accuracy*100):.2f}%')
    
`from sklearn.metrics import confusion_matrix`

`y_preds =np.round(model_9.predict(X_test))`

`confusion_matrix(y_test,y_preds)`

    # pretify the confusion matrix
    import itertools

    figsize = (10,10)

    # create the confusion matrix
    cm = confusion_matrix(y_test,y_preds)
    cm_norm = cm.astype("float") /cm.sum(axis = 1)[:,np.newaxis] # normalize our confusion matrix
    n_classes = cm.shape[0]

    # Let prettify it
    fig,ax = plt.subplots(figsize=figsize)
    # create a matrix plot
    cax = ax.matshow(cm,cmap = plt.cm.Blues)
    fig.colorbar(cax)

    # create classes
    classes = False

    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title = "Confusion Matrix",
          xlabel = "Predicted Label",
          ylabel = "True Label",
          xticks = np.arange(n_classes),
          yticks = np.arange(n_classes),
          xticklabels = labels,
          yticklabels = labels)

    # Set x-axis labels to bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Adjust label size
    ax.yaxis.label.set_size(20)
    ax.xaxis.label.set_size(20)
    ax.title.set_size(20)

    # Set threshold for different colors
    threshold = (cm.max() + cm.min())/2

    # Plot the text on each cell
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,f"{cm[i,j]} ({cm_norm[i,j]*100:.1f}%)",
                 horizontalalignment = 'center',
                 color = 'white' if cm[i,j] > threshold else 'black',
                 size = 15)

`from tensorflow.keras.datasets import fashion_mnist`

`(train_data,train_labels),(test_data,test_labels) = fashion_mnist.load_data()`

    # set the random seed
    tf.random.set_seed(42)
    # build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape = train_data[0].shape), # pass the image 28x28
        tf.keras.layers.Dense(4,activation = 'relu'),
        tf.keras.layers.Dense(4,activation = 'relu'),
        tf.keras.layers.Dense(10,activation = 'softmax')
    ])
    # compile the model
    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                 optimizer = tf.keras.optimizers.Adam(),
                 metrics = ['accuracy'])
    # fit the model
    non_norm_history = model.fit(train_data,
                                 train_labels,
                                 epochs =10,
                                 verbose =1,
                                 validation_data = (test_data,test_labels))
