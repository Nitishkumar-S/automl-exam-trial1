# class NeuralNetwork:
#     def __init__(self, input_shape, num_classes):
#         self.input_shape = input_shape
#         self.num_classes = num_classes
#         self.model = self.build_model()

#     def build_model(self):
#         from tensorflow.keras.models import Sequential
#         from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

#         model = Sequential()
#         model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Flatten())
#         model.add(Dense(128, activation='relu'))
#         model.add(Dense(self.num_classes, activation='softmax'))

#         model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#         return model

#     def train(self, x_train, y_train, epochs=10, batch_size=32, validation_data=None):
#         self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(config['dropout']),
            nn.Flatten(),
            nn.Linear(14 * 14 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)
