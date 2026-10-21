import numpy as np
import tensorflow as tf

models = [
    ("Baseline MLP INT8",
     "models/mfcc_mlp_int8.tflite",
     "models/mfcc_normalization.npz"),
    ("MLP v2 INT8",
     "models/mfcc_mlp_v2_int8.tflite",
     "models/mfcc_normalization_v2.npz"),
]

for name, model_path, norm_path in models:
    print("\n" + name)

    with np.load(norm_path) as norm:
        print("mean shape:", norm["mean"].shape)
        print("std shape:", norm["std"].shape)

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    print("input:", inp["shape"], inp["dtype"])
    print("output:", out["shape"], out["dtype"])
