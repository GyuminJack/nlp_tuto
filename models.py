import tensorflow as tf

class ConvModel(tf.keras.Model):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(20000, 256, trainable = True)
        self.conv1 = tf.keras.layers.Conv1D(128, 3, activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPooling1D()
        self.conv2 = tf.keras.layers.Conv1D(128, 4, activation='relu')
        self.maxpool2 = tf.keras.layers.GlobalMaxPooling1D()
        self.fc1 = tf.keras.layers.Dense(8, activation='relu')
        self.fc2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        embeddings = self.embedding(inputs)
        x = self.conv1(embeddings)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class BinaryModel(tf.keras.Model):
    def __init__(self):
        super(BinaryModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(20000, 256, trainable = True)
        self.lstm = tf.keras.layers.GRU(128)
        self.d1 = tf.keras.layers.Dense(32, activation='relu')
        self.d2 = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs):
        embeddings = self.embedding(inputs)
        x = self.lstm(embeddings)
        x = self.d1(x)
        x = self.d2(x)
        return x