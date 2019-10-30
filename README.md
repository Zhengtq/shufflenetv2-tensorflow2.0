# shufflenetv2-tensorflow2.0

```python
if __name__ == '__main__':                                       
                                                                 
    model = ShufflenetV2(num_classes=1, training=False)          
                                                                 
    x = tf.random.uniform((2, 224,224, 3))                       
    for ind in range(10000):                                     
        y = model(x)                                             
    model.build((1,224,224,3))                                   
    model.summary()                                              
```                                                          





