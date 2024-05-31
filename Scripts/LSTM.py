# LSTM

import tensorflow as tf
def lstm(train_generator,test_generator):


    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((224, 224*3), input_shape=(224, 224, 3)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    history = model.fit(
        train_generator,
        epochs=1,
        validation_data=test_generator,
        batch_size=32
    )