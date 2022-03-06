    import pathlib
    train_dir = pathlib.Path(dataset_name+'/train/')
    # Get the classes name of the image in train dataset
    train_classes = np.array(sorted([item.name for item in train_dir.glob('*')]))
    
`import matplotlib.image as mpimg`

`random_image = random.sample(os.listdir(target_folder),1)`

`img = mpimg.imread(target_folder + '/' + random_image[0])`

`from tensorflow.keras.preprocessing.image import ImageDataGenerator`

`train_datagen = ImageDataGenerator(rescale = 1./255)`

    train_data = train_datagen.flow_from_directory(train_dir,
                              batch_size = 32,
                              target_size = (224,224),
                              class_mode = 'categorical',
                              seed = 42
                              )
                              
`train_data.image_shape`

    train_datagen_augmented = ImageDataGenerator(rescale = 1./255,
                                                rotation_range = 20, # this is a int not a float
                                                width_shift_range = 0.2,
                                                height_shift_range = 0.2,
                                                zoom_range = 0.2,
                                                horizontal_flip = True
                                                )
                                                
`img = tf.io.read_file(filename)`

`img = tf.image.decode_image(img,channels = 3)`

`img = tf.image.resize(img,size = [img_shape,img_shape])`

`(X_train,y_train),(X_test,y_test) = tf.keras.datasets.mnist.load_data()`
