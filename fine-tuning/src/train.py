import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from funcy import take
from itertools import starmap


# データセットを作成します。
training_data = tf.keras.utils.image_dataset_from_directory('../input/cats_and_dogs_filtered/train', image_size=(224, 224))
validation_and_test_data = tf.keras.utils.image_dataset_from_directory('../input/cats_and_dogs_filtered/validation', image_size=(224, 224))
test_data = validation_and_test_data.take(int(validation_and_test_data.cardinality().numpy() * 0.2))
validation_data = validation_and_test_data.skip(int(validation_and_test_data.cardinality().numpy() * 0.2))

# 元になるモデルを取得します。
base_model = tf.keras.applications.mobilenet_v2.MobileNetV2()

# 元になるモデルから不要な層を削除した新しいモデルを作成します。
base_model = tf.keras.Model(base_model.layers[0].input, base_model.layers[-2].output, name=base_model.name)

# ニューラル・ネットワークを作成します。
input = base_model.input
x = input
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # 入力をMobileNetV2の要件に合わせます。
x = tf.keras.layers.RandomFlip('horizontal')(x)             # ランダムで画像の左右を反転します。
x = tf.keras.layers.RandomRotation(0.2)(x)                  # ランダムで画像を回転させます。
x = base_model(x, training=False)                           # 元になるモデルを呼び出します。BatchNormalizationとかでtrainable=Trueにすると問題がでるので、training=Falseを追加しておきます。
x = tf.keras.layers.Dense(2)(x)                             # 猫か犬かを分類する層を追加します。
output = x

# モデルを作成します。
model = tf.keras.Model(input, output)

# 元になるモデルの層を、学習しないように設定します。
for layer in base_model.layers[:]:
    layer.trainable = False

# モデルのサマリーを出力します。
model.summary()

# モデルを学習します。
model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=(tf.keras.metrics.SparseCategoricalAccuracy(),))
history_1 = model.fit(training_data, epochs=10, validation_data=validation_data)

# もう一回！

# 元になるモデルのうち、上の方の層を学習可能に設定します。
for layer in base_model.layers[-int(len(base_model.layers) * 0.25):]:
    layer.trainable = True
base_model.summary(show_trainable=True)

# モデルのサマリーを出力します。
model.summary()

# モデルを学習します。
model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.00001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=(tf.keras.metrics.SparseCategoricalAccuracy(),))
history_2 = model.fit(training_data, epochs=history_1.epoch[-1] + 1 + 10, initial_epoch=history_1.epoch[-1] + 1, validation_data=validation_data)

# 学習曲線を表示します。
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(history_1.history['loss'] + history_2.history['loss'], label='Training')
plt.plot(history_1.history['val_loss'] + history_2.history['val_loss'], label='Validation')
plt.legend(loc='upper right')
plt.ylim([0, max(history_1.history['val_loss'] + history_2.history['val_loss'])])
plt.title('Loss')
plt.subplot(2, 1, 2)
plt.plot(history_1.history['sparse_categorical_accuracy'] + history_2.history['sparse_categorical_accuracy'], label='Training')
plt.plot(history_1.history['val_sparse_categorical_accuracy'] + history_2.history['val_sparse_categorical_accuracy'], label='Validation')
plt.legend(loc='lower right')
plt.ylim([min(history_1.history['val_sparse_categorical_accuracy'] + history_2.history['val_sparse_categorical_accuracy']), 1])
plt.title('Accuracy')
plt.show()

# テスト・データでの精度を確認します。
ys, ys_pred = map(np.array, zip(*starmap(lambda xs, ys: (ys, model.predict_on_batch(xs)),
                                         test_data.as_numpy_iterator())))
metric = tf.keras.metrics.SparseCategoricalAccuracy()
metric.update_state(ys, ys_pred)
print(f"accuracy: {metric.result().numpy()}")

# テスト・データで予測してみます。
xs, ys = test_data.as_numpy_iterator().next()
ys_pred = np.argmax(model.predict_on_batch(xs), axis=1)  # 学習したモデルで予測をします。
print(f"ys     : {ys}")
print(f"ys_pred: {ys_pred}")

# テスト・データでの予測結果を可視化します。
plt.figure(figsize=(10, 10))
for i, [x, y, y_pred] in enumerate(take(9, zip(xs, ys, ys_pred))):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x.astype(np.uint8))
    plt.title(training_data.class_names[y_pred])
    plt.axis('off')
plt.show()

# モデルを保存します。
model.save('../working/model')
