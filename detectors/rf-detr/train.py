from rfdetr import RFDETRBase

model = RFDETRBase()

model.train(
    dataset_dir="/root/tungn197/license-plate-recognition/data/vehicle-detection-3",
    epochs=10,
    batch_size=16,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="./runs"
)