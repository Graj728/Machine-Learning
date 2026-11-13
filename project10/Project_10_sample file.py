"""Project 10: Introduction to Deep Learning — guided MNIST activity.
Run ONCE without changing the model settings.
if needed Install: python -m pip install tensorflow numpy matplotlib
The MNIST data downloads automatically on first use.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Preset configuration. We do not need to edit these settings.
EPOCHS = 3
TRAINING_EXAMPLES = 12000


def main():
    tf.keras.utils.set_random_seed(42)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    fig, axes = plt.subplots(2, 5, figsize=(9, 4))
    for digit, ax in enumerate(axes.flat):
        idx = np.flatnonzero(y_train == digit)[0]
        ax.imshow(x_train[idx], cmap='gray')
        ax.set_title(f'Digit {digit}')
        ax.axis('off')
    fig.tight_layout()
    fig.savefig('mnist_samples.png', dpi=150)
    plt.close(fig)

    x_train = x_train[:TRAINING_EXAMPLES].astype('float32') / 255.0
    y_train = y_train[:TRAINING_EXAMPLES]
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=64,
                        validation_split=0.10, verbose=2)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f'\nTEST ACCURACY: {accuracy:.2%}')

    fig, ax = plt.subplots(figsize=(7, 4))
    ep = range(1, EPOCHS + 1)
    ax.plot(ep, history.history['accuracy'], marker='o', label='Training accuracy')
    ax.plot(ep, history.history['val_accuracy'], marker='o', label='Validation accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and validation accuracy')
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig('training_accuracy.png', dpi=150)
    plt.close(fig)

    predicted = np.argmax(model.predict(x_test, batch_size=256, verbose=0), axis=1)
    wrong = np.flatnonzero(predicted != y_test)
    if len(wrong) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
        for ax, idx in zip(axes, wrong[:2]):
            ax.imshow(x_test[idx, :, :, 0], cmap='gray')
            ax.set_title(f'Actual: {y_test[idx]} | Predicted: {predicted[idx]}')
            ax.axis('off')
        fig.tight_layout()
        fig.savefig('misclassified_digits.png', dpi=150)
        plt.close(fig)
    print('Saved mnist_samples.png, training_accuracy.png, and misclassified_digits.png')

if __name__ == '__main__':
    main()
