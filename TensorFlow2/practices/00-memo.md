# 00_tensorflow_fundamentals (memo)

`tf.constant` `tf.Variable`
    
`tf.random.set_seed` `tf.random.Generator.from_seed(42)` 

`random.uniform` `random.normal`

`tf.random.shuffle`

`tf.random.zeros()` `tf.random.ones()`

`tensor.ndim` `tensor.shape` `tensor.dtype` `tf.size(tensor)` `tf.size(tensor).numpy()`

`tensor[]`

`rank_3_tensor = rank_2_tensor[...,tf.newaxis]` `tf.expand_dims(rank_2_tensor,axis = -1)`

`tf.math`

`tf.math.multiply(tensor,10)` `tf.linalg.matmul()` `tf.matmul()` `tensor1 @ tensor2` `tf.tensordot()`

`tf.reshape(tensor2, shape = (2,3))`

`tf.transpose(tensor)`

`tf.tensordot(tf.transpose(X),Y, axes =1)`

`B = tf.cast(B,dtype = tf.float16)`

`E = tf.abs(D)` `tf.reduce_min(E)` `tf.reduce_max(E)` `tf.reduce_mean(E)` `tf.reduce_sum(E)`

`!pip install tensorflow-propability`

`import tensorflow_probability as tfp`

`var = tfp.stats.variance(E)` `tf.math.reduce_std(tf.cast(E,dtype=tf.float32))`

`tf.argmax(F)` `tf.argmin(F)`

`tf.squeeze(G)` `tf.one_hot(some_list,depth = len(some_list))` `tf.one_hot(some_list, depth= 4, on_value = 'bar',off_value = 'foo')`

`tf.range(1,10)` `tf.sqrt(tf.cast(H,dtype = tf.float32))` `tf.square(H)` `tf.math.log(tf.cast(H,dtype = tf.float32))`

`j = np.array(J)` `j = J.numpy()`
