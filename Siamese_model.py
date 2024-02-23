import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.layers import Input, Lambda, Dense
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class SiameseTrainer:
    def __init__(self, base_model_func, input_shape, num_classes):
        self.base_model_func = base_model_func
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        self.history = None
        
    def euclidean_distance(self, vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))
    
    def eucl_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)
        
    def build_model(self):
        left_input = Input(self.input_shape)
        right_input = Input(self.input_shape)
        
        base_model = self.base_model_func(self.input_shape, self.num_classes)
        encoded_left = base_model(left_input)
        encoded_right = base_model(right_input)
        
        distance = Lambda(self.euclidean_distance, output_shape=self.eucl_dist_output_shape)([encoded_left, encoded_right])
        
        # Dense layer to generate the similarity score
        prediction = Dense(1, activation='sigmoid')(distance)
        
        # Connect the inputs with the outputs
        siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
        
        return siamese_net
    
    def compile_model(self, optimizer, loss):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        
    # def train_model(self, train_generator, val_generator, epochs, steps_per_epoch, validation_steps):
    #     self.history = self.model.fit_generator(train_generator, 
    #                                            validation_data=val_generator,
    #                                            epochs=epochs,
    #                                            steps_per_epoch=steps_per_epoch,
    #                                            validation_steps=validation_steps)
    
    def train_model(self, train_pairs, train_labels, val_pairs, val_labels, epochs, batch_size):
        self.history = self.model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_labels,
                                     validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_labels),
                                     epochs=epochs,
                                     batch_size=batch_size)
        
    def plot_training(self):
        if self.history is None:
            print("No training history available")
            return
        
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        
        epochs = range(len(acc))
        
        plt.figure(figsize=(12,4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'bo', label='Training accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        
        plt.show()
        
    def evaluate_model(self, test_pairs, test_labels):
        test_loss, test_acc = self.model.evaluate([test_pairs[:, 0], test_pairs[:, 1]], test_labels)
        print("Test accuracy: ", test_acc)
        
        
    def plot_confusion_matrix(self, test_pairs, test_labels, threshold=0.5):
        if self.history is None:
            print("No training history available")
            return
        
        y_pred = self.model.predict([test_pairs[:, 0], test_pairs[:, 1]])
        y_pred = (y_pred > threshold).astype(int)
        y_true = test_labels.astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.show()
        
    
        
        