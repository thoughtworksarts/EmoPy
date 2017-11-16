from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model


class TransferModel:

    def __init__(self, model_name='inception_v3'):
        self.model_name = model_name
        self.model = self.init_model()

    def init_model(self):

        # create the base pre-trained model
        base_model = None
        if self.model_name == 'inception_v3':
            base_model = InceptionV3(weights='imagenet', include_top=False)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        # add a logistic layer -- FER+ has 7 prediction classes
        # (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
        predictions = Dense(units=7, activation='softmax')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        print(model.summary())

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def fit(self, X_train, y_train, X_test, y_test):
        self.model.fit(x=X_train, y=y_train, epochs=50, verbose=1, callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3)], validation_data=(X_test, y_test), shuffle=True)

        for layer in self.model.layers[:249]:
            layer.trainable = False
        for layer in self.model.layers[249:]:
            layer.trainable = True

        self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(x=X_train, y=y_train, epochs=50, verbose=1, callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3)], validation_data=(X_test, y_test), shuffle=True)

