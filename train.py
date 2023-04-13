import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow import keras
from loss import loss, psnr
from model import build_model
from load_data import get_train_dataset, get_val_dataset

train_dataset = get_train_dataset()
val_dataset = get_val_dataset()
print(train_dataset)
print(val_dataset)

model = build_model()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=2e-4), loss=loss, metrics=[psnr]
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=500,
    callbacks=[
        keras.callbacks.ReduceLROnPlateau(
            monitor="psnr",
            factor=0.5,
            patience=5,
            min_delta=1e-7,
            mode="max",
            min_lr=1e-6,
        ),
        keras.callbacks.ModelCheckpoint(
			filepath="./checkpoint/",
            monitor="val_psnr",
            save_best_only=True,
            mode="max",
            save_weights_only=True,
		)
    ],
)