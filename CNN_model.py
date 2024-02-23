import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

class ModelTrainer:
    
    def __init__(self, model, train_csv, validation_csv, test_csv, image_dir, seed = None):
        self.model = model
        self.train_csv = train_csv
        self.validation_csv = validation_csv
        self.test_csv = test_csv
        self.image_dir = image_dir
        self.seed = seed
        self.train_datagen = None
        self.validation_datagen = None
        self.test_datagen = None
        
    def load_data(self):
        if self.seed:
            np.random.seed(self.seed)
        # Load train, validation, and test data from CSV files
        self.train_df = pd.read_csv(self.train_csv, header=None, names=["file_path"])
        self.validation_df = pd.read_csv(self.validation_csv, header=None, names=["file_path"])
        self.test_df = pd.read_csv(self.test_csv, header=None, names=["file_path"])

        # Extract category labels from file paths
        self.train_df['label'] = self.train_df['file_path'].apply(lambda x: x.split("\\")[0])
        self.validation_df['label'] = self.validation_df['file_path'].apply(lambda x: x.split("\\")[0])
        self.test_df['label'] = self.test_df['file_path'].apply(lambda x: x.split("\\")[0])

        # Shuffle the data
        self.train_df = shuffle(self.train_df)
        self.validation_df = shuffle(self.validation_df)
        self.test_df = shuffle(self.test_df)
        
    def compile_model(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss = loss, metrics=metrics)
        
    def train_model(self, batch_size, epochs, model_save_name):
        self.train_datagen = ImageDataGenerator(rescale=1./255)
        self.validation_datagen = ImageDataGenerator(rescale=1./255)
        self.test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = self.train_datagen.flow_from_dataframe(
            dataframe=self.train_df,
            directory=self.image_dir,
            x_col="file_path",
            y_col="label",
            target_size=(96, 96),
            color_mode="grayscale",
            batch_size=batch_size,
            class_mode='sparse')

        validation_generator = self.validation_datagen.flow_from_dataframe(
            dataframe=self.validation_df,
            directory=self.image_dir,
            x_col="file_path",
            y_col="label",
            target_size=(96, 96),
            color_mode="grayscale",
            batch_size=batch_size,
            class_mode='sparse')

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

        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=self.train_df.shape[0] // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=self.validation_df.shape[0] // batch_size,
            callbacks=[model_checkpoint_callback])

        # Check if the last model had the best validation accuracy
        final_val_accuracy = self.history.history['val_accuracy'][-1]
        if final_val_accuracy > best_val_accuracy:
            best_val_accuracy = final_val_accuracy
            best_model_path = model_save_name
            self.model.save(best_model_path)
            print(f"New best model with validation accuracy {best_val_accuracy} saved as {best_model_path}")
        else:
            print(f"No new best model found. Best validation accuracy remains {best_val_accuracy}.")
            
    
    # def train_model(self, batch_size, epochs, model_save_name):
    #     # global train_datagen, validation_datagen, test_datagen
    #     self.train_datagen = ImageDataGenerator(rescale=1./255)
    #     self.validation_datagen = ImageDataGenerator(rescale=1./255)
    #     self.test_datagen = ImageDataGenerator(rescale=1./255)

    #     train_generator = self.train_datagen.flow_from_dataframe(
    #         dataframe=self.train_df,
    #         directory=self.image_dir,
    #         x_col="file_path",
    #         y_col="label",
    #         target_size=(96, 96),
    #         color_mode="grayscale",
    #         batch_size=batch_size,
    #         class_mode='sparse')

    #     validation_generator = self.validation_datagen.flow_from_dataframe(
    #         dataframe=self.validation_df,
    #         directory=self.image_dir,
    #         x_col="file_path",
    #         y_col="label",
    #         target_size=(96, 96),
    #         color_mode="grayscale",
    #         batch_size=batch_size,
    #         class_mode='sparse')
        
    #     # Define the checkpoint to save the best model
    #     model_checkpoint_callback = ModelCheckpoint(
    #         filepath=model_save_name,  # File path to save the model
    #         save_best_only=True,  # Only save the best model
    #         monitor='val_accuracy',  # Monitor validation accuracy
    #         mode='max')  # Save the model with max validation accuracy

    #     self.history = self.model.fit(
    #         train_generator,
    #         steps_per_epoch=self.train_df.shape[0] // batch_size,
    #         epochs=epochs,
    #         validation_data=validation_generator,
    #         validation_steps=self.validation_df.shape[0] // batch_size,
    #         callbacks=[model_checkpoint_callback])
        
        
        
    #     # After training, save the final model regardless of its performance
    #     final_model_path = "final_" + model_save_name  # Prefixing with 'final_' to differentiate
    #     self.model.save(final_model_path)
    #     print(f"Final model saved as {final_model_path}")
        
        
    def evaluate_model(self, batch_size):
        global test_generator
        test_generator = self.test_datagen.flow_from_dataframe(
            dataframe=self.test_df,
            directory=self.image_dir,
            x_col="file_path",
            y_col="label",
            target_size=(96, 96),
            color_mode="grayscale",
            batch_size=batch_size,
            class_mode='sparse')

        test_loss, test_acc = self.model.evaluate(test_generator, steps=self.test_df.shape[0] // batch_size)
        print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")


        
    def plot_training_history(self, epochs):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
        
    # def plot_confusion_matrix(self):
    #     y_pred = self.model.predict(test_generator)
    #     y_pred = np.argmax(y_pred, axis=1)
    #     y_true = test_generator.classes
    #     print(test_generator.classes)
    #     print(np.unique(test_generator.classes, return_counts=True))
    #     y_pred_classes = np.argmax(y_pred, axis=1)
    #     print("Predicted classes:", y_pred_classes)
    #     print("True classes:", y_true)
    #     print("Unique predicted classes:", np.unique(y_pred_classes))
    #     class_names = list(test_generator.class_indices.keys())
        
    #     cm = confusion_matrix(y_true, y_pred)
    #     cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize confusion matrix

    #     plt.figure(figsize=(8, 6))
    #     sns.heatmap(cm_normalized, annot=cm, cmap='Blues', fmt='d', xticklabels=class_names, yticklabels=class_names)
    #     plt.xlabel('Predicted labels')
    #     plt.ylabel('True labels')
    #     plt.title('Confusion Matrix')
    #     plt.show()
        
    def plot_confusion_matrix(self):
        y_pred = self.model.predict(test_generator)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = test_generator.classes
        class_names = list(test_generator.class_indices.keys())
        
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=False, cmap='Blues', fmt='.2f', xticklabels=class_names, yticklabels=class_names)

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = plt.text(j + 0.5, i + 0.5, f'{cm_normalized[i, j] * 100:.2f}%',
                                ha='center', va='center', color='black')

        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.show()


        
    # def plot_confusion_matrix(self):
    #     y_pred = self.model.predict(test_generator)
    #     y_pred = np.argmax(y_pred, axis=1)
    #     # cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     y_true = test_generator.classes
    #     class_names = list(test_generator.class_indices.keys())
        
    #     cm = confusion_matrix(y_true, y_pred)
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize confusion matrix
        
    #     plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    #     sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', xticklabels=class_names, yticklabels=class_names)
    #     plt.xlabel('Predicted labels')
    #     plt.ylabel('True labels')
    #     plt.title('Confusion Matrix')
    #     plt.show()
        
