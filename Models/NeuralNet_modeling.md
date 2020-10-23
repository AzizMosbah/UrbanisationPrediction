---
jupyter:
  jupytext:
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: bainhack
    language: python
    name: bainhack
---

# Convolutional Neural Net as a Regressor
    1. Try it on one tile at a time 
    2. Try it on one tile and 1 or 2 neighbors
    3. Try it by adding the neighbors outcome to the fully connected layer

```python
import os
os.chdir('../')
```

```python
import tensorflow as tf 
import numpy as np
from Pipelines.transformations import pixel_for_NN, split
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
```

    Note: pixel_for_NN takes at least 20 minutes to run 

```python
df = pixel_for_NN()
```

```python
df.head()
```

```python
l = []
for index, row in df.iterrows():
    l.append(np.stack((row.pixels, row.neighbor1, row.neighbor2, row.neighbor3), axis =0))
```

```python
full_data = np.stack(l, axis=0)
targets = np.log(1+df['target'])
```

```python
full_data = full_data.reshape(104835, 4, 33, 33, 1)
```

```python
full_data.shape
```

```python
X_train, X_test, y_train, y_test = train_test_split(full_data, targets, test_size=0.1, random_state=42)
```

```python
X_train.shape
```

```python
def build_cnn_model():
    cann_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(input_shape = (4, 33,33,1), filters = 24, kernel_size = (2,2)), 
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation=tf.nn.relu),
        tf.keras.layers.Dense(10,activation=tf.nn.relu),
        tf.keras.layers.Dense(1)
    ])
    
    return cann_model
  
cnn_model = build_cnn_model()

cnn_model.predict(X_train[[0]])

print(cnn_model.summary())
```

```python
cnn_model.compile(optimizer= tf.optimizers.Adam(learning_rate=.0001), loss='mean_squared_error', metrics=['mse'])
```

```python
history = cnn_model.fit(X_train, y_train, epochs = 10)
```

```python
test_loss, test_acc = cnn_model.evaluate(X_test, y_test)

print('Test accuracy:', test_acc)
```

```python

```
