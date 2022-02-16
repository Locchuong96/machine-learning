`from sklearn.datasets import make_circles`

`X,y = make_circles(n_samples,noise = 0.03, random_state = 42)`

`circles = pd.DataFrame({"X0":X[:,0],"X1":X[:,1],"label":y})`

`plt.scatter(X[:,0],X[:,1],c=y, cmap =plt.cm.RdYlBu)`

    `
    model_2.compile(loss = tf.keras.losses.BinaryCrossentropy(),
                   optimizer = tf.keras.optimizers.SGD(),
                   metrics = ['accuracy'])
    `
