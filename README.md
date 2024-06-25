# Seeds-Dataset-Multiclass-Classification

![multiclass](https://github.com/radiadus/Seeds-Dataset-Multiclass-Classification/assets/55176713/498f367e-3644-442e-b5c1-0dab19655aa0)

For this project I used pandas to read csv file, sklearn to preprocess the data and tensorflow 1.x to train the model. The seeds dataset is 210 instances for 3 kinds of wheat seed. You can get the dataset from kaggle in the this link: https://www.kaggle.com/rwzhang/seeds-dataset.

To start this project, I need to import some libraries.

    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    from sklearn.model_selection import train_test_split
    import tensorflow as tf

Then define functions for load data and preprocess data.

    # load data CSV
    def load_data():
        data = pd.read_csv('seeds_dataset.csv', names=['a','b','c','d','e','f','g','t'])
        features = data[['a','b','c','d','e','f','g']]
        target = data[['t']]
        
        return features, target

    # Pre-processing data
    def preprocess_data(features, target):
        features = MinMaxScaler().fit_transform(features)
        target = OneHotEncoder(sparse=False).fit_transform(target)
        
        return features, target

After that, I need to define how many nodes I wanted for our architecture and also initialized random weight and bias for the layers.

    # Inisialisasi layers, weights, dan bias
    layers = {
        'input' : 7 ,
        'hidden' : 7 , # 2 / 3 dari input + outputnya
        'output' : 3,
    }
    
    weights = {
        'hidden' : tf.Variable( tf.random.normal( [layers['input'], layers['hidden']] ) ),
        'output' : tf.Variable( tf.random.normal( [layers['hidden'], layers['output']] ) ),
    }
    
    bias = {
        'hidden' : tf.Variable( tf.random.normal( [layers['hidden'] ])),
        'output' : tf.Variable( tf.random.normal( [layers['output'] ])),
    }

Then I will need to define our activation function. In this case I was using sigmoid. Also I need to define a function for our feed forward process. The function will do matrix multiplication for our features and weights and add the result with the bias. The result will be activated with activation function then I will do multiplication and activation again until I reached the output layer.

    # Membuat placeholder untuk features dan target
    features_container = tf.placeholder(tf.float32, [None, layers['input']])
    target_container = tf.placeholder(tf.float32, [None, layers['output']]) #(?, 3)
    
    # Membuat fungsi menghitung prediksi dan menghitung error
    predicted_target = feed_forward(features_container)
    error = tf.reduce_mean(0.5 * (target_container - predicted_target)**2 )
    
    # Inisialisasi learning rate dan error
    learning_rate = 0.2
    epoch = 5000
    
    # Membuat fungsi training dengan menggunakan GradientDescentOptizer
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

The code above is used to make placeholder for our further process, initialized variables to predict target, calculate error, learning rate, epochs, and calculate model training. After that lets load and preprocess the data. Then split them to training dataset and testing dataset.

    # Data Processing
    features, target = load_data()
    features, target = preprocess_data(features, target)
    
    feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.2)

Last step and the most important part, run our training model with tensorflow and print the result.

    with tf.Session() as sess:
        # Inisialisasi variabel-variabel yang telah dibuat secara global
        sess.run(tf.global_variables_initializer())
    
        for i in range(epoch):
            # Menaruh feature dan target training ke dalam placeholder
            train_data = {
                features_container: feature_train,
                target_container: target_train,
            }
    
            # Menjalankan training dan menghitung loss
            sess.run(train, feed_dict = train_data)
            loss = sess.run(error, feed_dict = train_data)
    
            # Print loss setiap 500 epoch
            if i % 500 ==  0:
                print(f'Epoch {i}, Loss Rate: {loss}')
    
        # Menaruh feature dan target testing ke dalam placeholder
        test_data = {
            features_container: feature_test,
            target_container: target_test,
        }
    
        # Menghitung tingkat akurasi dan menampilkan confusion matrix
        accuracy = tf.equal(tf.argmax(target_container, axis = 1), tf.argmax(predicted_target, axis = 1) )
        print(f'Accuracy Row : {sess.run(accuracy, feed_dict = test_data)}')
        accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
        print(f'Final Accuracy : {sess.run(accuracy, feed_dict = test_data) * 100}%')
        conf = tf.math.confusion_matrix(tf.argmax(target_container, axis = 1), tf.argmax(predicted_target, axis = 1) )
        print('Confusion Matrix:')
        print(sess.run(conf, feed_dict = test_data))
        
![image-15](https://github.com/radiadus/Seeds-Dataset-Multiclass-Classification/assets/55176713/d3bdcce0-5a81-4dee-88b5-bf5744283220)

