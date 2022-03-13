    !tensorboard dev upload --logdir ./tensorflow_hub \
      --name 'efficientnetB0 vs resnet50v2' \
      --description 'Comparing two different TF hub feature extraction models architectures using 10 % training images' \
      --one_shot
      
    def create_model(model_url,num_classes = 10):
      '''
      Take a TensorFlow Hub URL and creates a Keras Sequential model with it
      Arguments:
      model_url --- the url of model (a tensorflow hub feature extraction URL)
      num_classes --- the number of output, should be equal to number of target classes, default 10
      Return:
      An uncompiled Keras Sequential model with model_url as featrue extractor layer and Dense ouput layer
      with num_classes outputs
      '''
      # Download the pretrained model and save it as Keras layer
      feature_extractor_layer = hub.KerasLayer(model_url,
                                               trainable = False, # Freeze the underlying patterns
                                               name = 'feature_extraction_layer',
                                               input_shape  = IMAGE_SHAPE + (3,) # define the input image shape
                                               )
      # Create our own model
      model = tf.keras.Sequential([
                                   feature_extractor_layer, # use the feature extractor as a base
                                   layers.Dense(num_classes,activation = 'softmax', name = 'output_layer') #create our own output layer
                                  ])
      return model
  
`!tensorboard dev list`

    def create_tensorboard_callback(dir_name,experiment_name):
      '''
      Function to create tensorboard callback
      Aguments:
      dir_name --- directory to save callback
      experiment_name --- the name of your experiment
      Return:
      tensorboard_callback --- the callback of tensorboard
      '''
      log_dir = dir_name + '/' + experiment_name + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
      tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir)
      print(f'Saving TensorBoard log files to: {log_dir}')
      return tensorboard_callback
  
`!tensorboard dev delete --experiment_id uvACDjzlRyiDM3il91fb6A`

    %reload_ext tensorboard
    %load_ext tensorboard
    %tensorboard --logdir ./tensorflow_hub/efficientnetB0/20220306-105449

`tensorflow_dataset.load()`

`tf.keras.layers.Resizing`

`tf.keras.layers.Rescaling` 

`tf.keras.layers.RandomFlip`

`tf.keras.layers.RandomRotation`

`tf.image.flip_left_right`

`tf.image.rgb_to_grayscale`

`tf.image.adjust_brightness`

`tf.image.central_crop`

`tf.image.stateless_random*`
