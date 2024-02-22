from CNN_model import ModelTrainer
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import os

print("chemin", os.getcwd() + "/SiameseNetwork")
print("CHEMIN", os.path.join(os.path.dirname(os.getcwd())), "SiameseNetwork")
base_path = os.path.join(os.path.dirname(os.getcwd()),"Projet_systeme_3A", "SiameseNetwork", "Split_Tanker_Bulk_Container_frugal_vv")

# Create a CNNModel object
def simple_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(1024, activation='relu'),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

input_shape = (96, 96, 1)
num_classes = 3
model = simple_cnn(input_shape, num_classes)
optimizer = Adam(learning_rate=0.001)
loss = 'sparse_categorical_crossentropy'
batch_size = 32

# base_path = os.path.join(os.path.dirname(os.getcwd()), "SiameseNetwork", "Split_Tanker_Bulk_Container_frugal_vv")
# experiment = ModelTrainer(model, 'Split_Tanker_Bulk_Container_frugal_vv/train.csv', 'Split_Tanker_Bulk_Container_frugal_vv/validation.csv', 'Split_Tanker_Bulk_Container_frugal_vv/test.csv', '../OpenSARShip/Categories/', seed=42)
# experiment = ModelTrainer(model, os.path.join(base_path, "train.csv"), os.path.join(base_path, "validation.csv"), os.path.join(base_path, "test.csv"), '../OpenSARShip/Categories/', seed=42)
# experiment = ModelTrainer(model, "../Split_Tanker_Bulk_Container_frugal_vv/train.csv", "../Split_Tanker_Bulk_Container_frugal_vv/validation.csv", "../Split_Tanker_Bulk_Container_frugal_vv/test.csv", '../OpenSARShip/Categories/', seed=42)
experiment = ModelTrainer(model, os.getcwd() + "/SiameseNetwork/Split_Tanker_Bulk_Container_frugal_vv/train.csv", os.getcwd() + "/SiameseNetwork/Split_Tanker_Bulk_Container_frugal_vv/validation.csv", os.getcwd() + "/SiameseNetwork/Split_Tanker_Bulk_Container_frugal_vv/test.csv", os.getcwd() + "/OpenSARShip/Categories/", seed=42)


experiment.load_data()
experiment.compile_model(optimizer, loss, ['accuracy'])
experiment.train_model(batch_size, epochs=10)
experiment.evaluate_model()
