import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ModelTrainer:
    
    def __init__(self, model, train_csv, validation_csv, test_csv, image_dir, seed = None):
        self.model = model
        self.train_csv = train_csv
        self.validation_csv = validation_csv
        self.test_csv = test_csv
        self.image_dir = image_dir
        self.seed = seed
        
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
        
    def train_model(self, batch_size, epochs):
        train_datagen = ImageDataGenerator(rescale=1./255)
        validation_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_dataframe(
            dataframe=self.train_df,
            directory=self.image_dir,
            x_col="file_path",
            y_col="label",
            target_size=(96, 96),
            color_mode="grayscale",
            batch_size=batch_size,
            class_mode='sparse')

        validation_generator = validation_datagen.flow_from_dataframe(
            dataframe=self.validation_df,
            directory=self.image_dir,
            x_col="file_path",
            y_col="label",
            target_size=(96, 96),
            color_mode="grayscale",
            batch_size=batch_size,
            class_mode='sparse')

        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=self.train_df.shape[0] // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=self.validation_df.shape[0] // batch_size)
        
    def evaluate_model(self):
        test_generator = test_datagen.flow_from_dataframe(
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
        
    def plot_training_history(self):
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
        
    def plot_confusion_matrix(self):
        y_pred = self.model.predict(test_generator)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = test_generator.classes
        class_names = list(test_generator.class_indices.keys())
        
        cm = confusion_matrix(y_true, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize confusion matrix
        
        plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.show()