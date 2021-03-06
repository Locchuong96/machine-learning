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

`model.input` `model.inputs`

`model.output` `model.outputs`

`model.variables` `model.weights`

`model.predict(tf.constant([17.0]),verbose = 1)`

`model.evaluate(tf.expand_dims(X_test,axis = -1),y_test,verbose =1)`

`plot_model(model, show_shapes = True)`

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

`model.input` `model.inputs`

`model.output` `model.outputs`

`model.variables` `model.weights`

`model.predict(tf.constant([17.0]),verbose = 1)`

`model.evaluate(tf.expand_dims(X_test,axis = -1),y_test,verbose =1)`

`plot_model(model, show_shapes = True)`

`tf.keras.losses.mean_absolute_error(y_test,tf.squeeze(y_pred))`

`mse = tf.losses.MSE(y_true= y_test,y_pred = tf.squeeze(y_pred))`

`model_2.save("best_model_SavedModel_format")` `model_2.save("best_model_HDF5_format.h5")`

`model = tf.keras.models.load_model('./best_model_SavedModel_format')` `model = tf.keras.models.load_model("./best_model_HDF5_format.h5")`

`insurance_one_hot = pd.get_dummies(insurance)` `insurance_one_hot = insurance_one_hot.drop(columns= ['sex_male','smoker_yes'])`

`X = insurance_one_hot.drop("charges",axis = 1)` `y = insurance['charges']` 
`X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)`

`history = insurance_model_3.fit(X_train,y_train,epochs = 200)`

`X['bmi'].plot(kind = 'hist')`

    `
    from sklearn.compose import make_column_transformer
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    from sklearn.model_selection import train_test_split

    # Create a column transformer to normalize
    ct = make_column_transformer(
                                (MinMaxScaler(),['age','bmi','children']), # turn all value in these columns between 0 and 1
                                (OneHotEncoder(handle_unknown= "ignore"),["sex","smoker","region"])
                                )

    # create X & y 
    X = insurance.drop("charges",axis = 1)
    y =insurance['charges']

    # build our train and test sets
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

    # fit the column transformer to or training data
    ct.fit(X_train)

    # Transform training and test data with normalization (MinMaxScaler) and OneHotEncoder
    X_train_normal = ct.transform(X_train)
    X_test_normal = ct.transform(X_test)
    `
    
`df.dtypes` `df.describe()`
