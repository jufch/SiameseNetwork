import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
# from tensorflow.layers import Input, Lambda, Dense
from tensorflow.keras.layers import Input, Lambda, Dense
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau



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
    
    def train_model(self, train_pairs, train_labels, val_pairs, val_labels, epochs, batch_size, model_save_name, callbacks=None):

        # Initialize variable to track the best validation accuracy
        best_val_accuracy = 0.0
        best_model_path = ""

        # Define the checkpoint to save the best model
        model_checkpoint_callback = ModelCheckpoint(
            filepath=model_save_name,  # File path to save the model
            save_best_only=True,  # Only save the best model
            monitor='val_accuracy',  # Monitor validation accuracy
            mode='max',  # Save the model with max validation accuracy
            verbose=1)  # Log when a model is being saved
        
        # Define early stopping callback
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',  # Monitor the validation loss
            patience=5,  # Number of epochs with no improvement after which training will be stopped
            restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored quantity
            verbose=1)
        
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, mode = 'min')
        
        self.history = self.model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_labels,
                                     validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_labels),
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     callbacks = [model_checkpoint_callback, reduce_lr])
        
        # Check if the last model had the best validation accuracy
        final_val_accuracy = self.history.history['val_accuracy'][-1]
        if final_val_accuracy > best_val_accuracy:
            best_val_accuracy = final_val_accuracy
            best_model_path = model_save_name
            self.model.save(best_model_path)
            print(f"New best model with validation accuracy {best_val_accuracy} saved as {best_model_path}")
        else:
            print(f"No new best model found. Best validation accuracy remains {best_val_accuracy}.")
        
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
        plt.plot(epochs, acc, label='Training accuracy')
        plt.plot(epochs, val_acc, label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, label='Training loss')
        plt.plot(epochs, val_loss, label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.show()
        
    def evaluate_model(self, test_pairs, test_labels):
        test_loss, test_acc = self.model.evaluate([test_pairs[:, 0], test_pairs[:, 1]], test_labels)
        print("Test accuracy: ", test_acc)
        
        
    # def plot_confusion_matrix(self, test_pairs, test_labels, threshold=0.5):
    #     if self.history is None:
    #         print("No training history available")
    #         return
        
    #     y_pred = self.model.predict([test_pairs[:, 0], test_pairs[:, 1]])
    #     y_pred = (y_pred > threshold).astype(int)
    #     y_true = test_labels.astype(int)
        
    #     cm = confusion_matrix(y_true, y_pred)
    #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    #     plt.xlabel('Predicted label')
    #     plt.ylabel('True label')
    #     plt.show()
        
    def plot_confusion_matrix_siamese(test_pairs, test_labels, model, threshold=0.5):
    # Predict the similarity for the test pairs
        y_pred = model.predict([test_pairs[:, 0], test_pairs[:, 1]])
        y_pred = (y_pred > threshold).astype(int)
        y_true = test_labels.astype(int)
        
        # Calculate the confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Define class names for binary classification
        class_names = ['Dissimilar', 'Similar']
        
        # Plotting
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=False, cmap='Blues', fmt='.2f', xticklabels=class_names, yticklabels=class_names)
        
        # Adding text annotation for percentages
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                percentage = f'{cm_normalized[i, j] * 100:.2f}%'
                plt.text(j + 0.5, i + 0.5, percentage, ha='center', va='center', color='black')
        
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.show()
        
    def predict(self, image1, image2):
        img1 = load_img(image1, target_size=self.input_shape)
        img2 = load_img(image2, target_size=self.input_shape)
        
        img1 = img_to_array(img1)
        img2 = img_to_array(img2)
        
        img1 = img1 / 255.0
        img2 = img2 / 255.0
        
        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)
        
        similarity = self.model.predict([img1, img2])
        return similarity
    
    def evaluate_classification(self, test_data, reference_images, threshold=0.5):
    # test_data: list of tuples (image, true_label)
    # reference_images: dict with class names as keys and lists of reference image tensors as values
    
        predicted_labels = []
        true_labels = []
        
        for image, true_label in test_data:
            similarities = []
            for class_name, refs in reference_images.items():
                # Calculate similarity with reference images for each class
                class_similarity = np.mean([
                    self.model.predict([image, ref_image])[0][0] for ref_image in refs
                ])
                similarities.append((class_name, class_similarity))
            
            # Classify the image to the class with the highest similarity
            predicted_class = max(similarities, key=lambda x: x[1])[0]
            predicted_labels.append(predicted_class)
            true_labels.append(true_label)
        
        # Calculate the confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=list(reference_images.keys()))
        
        # Plot the confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=reference_images.keys(), yticklabels=reference_images.keys())
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.show()
        
        # Print the classification report
        print(classification_report(true_labels, predicted_labels, target_names=reference_images.keys()))
        
        return cm
    
        
        