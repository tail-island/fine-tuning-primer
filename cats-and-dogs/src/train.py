import tensorflow as tf

# データセットを作成します。
train_dataset = tf.keras.utils.image_dataset_from_directory('../input/cats_and_dogs_filtered/train', image_size=(224, 224))
validation_dataset = tf.keras.utils.image_dataset_from_directory('../input/cats_and_dogs_filtered/validation', image_size=(224, 224))

# 元になるモデルを取得します。
base_model = tf.keras.applications.mobilenet_v2.MobileNetV2()
# base_model.summary()

# 元になるモデルから不要な層を削除した新しいモデルを作成します。
base_model = tf.keras.Model(base_model.layers[0].input, base_model.layers[-2].output, name=base_model.name)
# base_model.summary()

# 元になるモデルのうち、学習しない範囲を設定します。
for layer in base_model.layers[:]:
    layer.trainable = False
# base_model.summary(show_trainable=True)

# ニューラル・ネットワークを作成します。
input = base_model.input
x = input
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # 入力をMobileNetV2の要件に合わせます。
x = tf.keras.layers.RandomFlip('horizontal')(x)             # ランダムで画像の左右を反転します。
x = tf.keras.layers.RandomRotation(0.2)(x)                  # ランダムで画像を回転させます。
x = base_model(x)
x = tf.keras.layers.Dense(2)(x)                             # 猫か犬かを分類する層を追加します。
output = x

# モデルを作成します。
model = tf.keras.Model(input, output)
model.summary(show_trainable=True)

# モデルを学習します。
model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0005),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=(tf.keras.metrics.SparseCategoricalAccuracy(),))
history = model.fit(train_dataset, epochs=10, validation_data=validation_dataset)
