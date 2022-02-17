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
