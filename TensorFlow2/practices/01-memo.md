# 01_neural_network_regression

`input_shape = X[0].shape` `output_shape = y[0].shape`

`model = tf.keras.Sequential()`

`input_shape = X[0].shape` `output_shape = y[0].shape`

`tf.keras.layers.Dense(1,input_shape = X[0].shape)`

`model.add(tf.keras.layers.Dense(1,input_shape = input_shape)`

    model = tf.keras.Sequential([
    
        tf.keras.layers.Dense(10,input_shape = [1],name= 'input_layer'),

        tf.keras.layers.Dense(1,name= 'output_layer')],
                              
        name = "neural_network")
        
        
`model.fit(tf.expand_dims(X,axis = -1),y,epochs = 5)`
    
    # expand feature's dim
    X = tf.expand_dims(X,axis = -1)

    # set random seed
    tf.random.set_seed(42)
    # get input shape and output shape
    input_shape = X[0].shape # the shape of 1 input sample
    output_shape = y[0].shape # the shape of 1 output sample

    print(f'input_shape {input_shape},output_shape {output_shape}')

    # create a model using Sequential API
    model = tf.keras.Sequential([tf.keras.layers.Dense(1,input_shape = X[0].shape)]) # input layer with 1 node

    # compile the model
    model.compile(
            loss = tf.keras.losses.mae,
            optimizer = tf.keras.optimizers.SGD(),
            metrics = ['mae']
            )

`model.compile(loss = , optimizer = ,metrics = )` `tf.keras.optimizers.Adam(learning_rate = 0.0001)`

`model.compile(loss = "mae", optimizer =tf.keras.optimizers.SGD(), metrics = ['mae'])`

`model.fit(tf.expand_dims(X, axis=-1), y, epochs=5)`

`model.layers`

`model.summary()`

`model.input, model.inputs`

`model.output,model.outputs`

`model.variables`

`model.weights`

`model.predict(tf.constant([17.0]),verbose = 1)`

`model.evaluate(tf.expand_dims(X_test,axis = -1),y_test,verbose =1)`

`plot_model(model, show_shapes = True)`
