# Person Detection Model Training for Arduino Nano BLE 33
# Undergraduate Research Symposium Project

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, Reshape, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import time

# Check if GPU is available
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# 1. Download and prepare a smaller dataset
# Using the Pascal VOC dataset - specifically extracting only person class
# Or you can use OpenImages V6 with person class only

# Sample code for downloading OpenImages V6 person class only
!pip install -q fiftyone
import fiftyone as fo
import fiftyone.zoo as foz

# Download only person class data from OpenImages V6
# This will be much smaller than COCO
dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="train",
    label_types=["detections"],
    classes=["Person"],
    max_samples=2000  # Limit to 2000 samples to keep it manageable
)

# Convert to TensorFlow format
# This is a simplified example - you'll need to adapt it for your specific dataset format
def prepare_dataset(dataset):
    # Extract images and annotations
    images = []
    labels = []
    
    for sample in dataset:
        img = tf.io.read_file(sample.filepath)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = img / 255.0  # Normalize
        
        # Extract bounding boxes for persons
        boxes = []
        for detection in sample.detections:
            if detection.label == "Person":
                boxes.append([
                    detection.bounding_box[1],  # ymin
                    detection.bounding_box[0],  # xmin
                    detection.bounding_box[3],  # ymax
                    detection.bounding_box[2],  # xmax
                ])
        
        images.append(img)
        labels.append(boxes)
    
    return tf.data.Dataset.from_tensor_slices((images, labels))

# 2. Create a MobileNetV2 base model with reduced input size for Arduino compatibility
def create_person_detection_model():
    # Use MobileNetV2 as base model with smaller input size
    base_model = MobileNetV2(
        input_shape=(96, 96, 3),  # Smaller input size
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add detection head
    x = base_model.output
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(5, (1, 1), padding='same')(x)  # 5 = [class, x, y, width, height]
    
    # Reshape output for bounding box prediction
    shapes = tf.shape(base_model.input)
    output = Reshape((-1, 5))(x)  # Reshape to [batch, num_boxes, 5]
    
    model = Model(inputs=base_model.input, outputs=output)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',  # Mean squared error for bounding box regression
        metrics=['accuracy']
    )
    
    return model

# 3. Train the model
model = create_person_detection_model()
print(model.summary())

# Callbacks for training
checkpoint = ModelCheckpoint(
    'person_detection_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1
)

# Training parameters
BATCH_SIZE = 16
EPOCHS = 10

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, early_stopping]
)

# 4. Evaluate the model and collect metrics
def evaluate_model(model, test_dataset):
    # Evaluate accuracy
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")
    
    # Measure inference time on GPU
    start_time = time.time()
    
    # Perform inference on a batch of images
    for batch in test_dataset.take(10):
        model.predict(batch[0])
    
    end_time = time.time()
    avg_inference_time = (end_time - start_time) / 10
    print(f"Average GPU Inference Time: {avg_inference_time * 1000:.2f} ms")
    
    return {
        'accuracy': test_accuracy,
        'inference_time': avg_inference_time
    }

# Collect GPU metrics
gpu_metrics = evaluate_model(model, test_dataset)

# 5. Convert the model to TensorFlow Lite for Arduino deployment
def convert_to_tflite(model):
    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Specify target hardware (enable when deploying)
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Quantize the model to reduce size
    # converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.uint8
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open('person_detection_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("Model converted to TensorFlow Lite successfully")
    print(f"TFLite Model Size: {len(tflite_model) / 1024:.2f} KB")
    
    return tflite_model

# Convert to TFLite
tflite_model = convert_to_tflite(model)

# 6. Test TFLite model inference on Colab
def test_tflite_model(tflite_model, test_dataset):
    # Initialize the TFLite interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Test inference time
    start_time = time.time()
    
    # Perform inference on sample images
    for batch in test_dataset.take(10):
        sample_input = batch[0][0].numpy()
        interpreter.set_tensor(input_details[0]['index'], [sample_input])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
    
    end_time = time.time()
    avg_inference_time = (end_time - start_time) / 10
    print(f"Average TFLite Inference Time: {avg_inference_time * 1000:.2f} ms")
    
    return avg_inference_time

# Test TFLite model
tflite_inference_time = test_tflite_model(tflite_model, test_dataset)

# 7. Visualization of results for poster
def visualize_metrics(gpu_metrics, tflite_inference_time):
    # Bar chart for inference time comparison
    plt.figure(figsize=(10, 6))
    plt.bar(['GPU', 'TFLite'], 
            [gpu_metrics['inference_time'] * 1000, tflite_inference_time * 1000])
    plt.title('Inference Time Comparison')
    plt.ylabel('Inference Time (ms)')
    plt.savefig('inference_time_comparison.png')
    plt.show()
    
    # Visualization of model architecture
    tf.keras.utils.plot_model(model, to_file='model_architecture.png', show_shapes=True)
    
    # Add model size comparison
    model_size_kb = os.path.getsize('person_detection_model.h5') / 1024
    tflite_size_kb = os.path.getsize('person_detection_model.tflite') / 1024
    
    plt.figure(figsize=(10, 6))
    plt.bar(['Full Model', 'TFLite Model'], [model_size_kb, tflite_size_kb])
    plt.title('Model Size Comparison')
    plt.ylabel('Size (KB)')
    plt.savefig('model_size_comparison.png')
    plt.show()

# Create visualizations
visualize_metrics(gpu_metrics, tflite_inference_time)

# Save the TFLite model to Google Drive for Arduino deployment
from google.colab import drive
drive.mount('/content/drive')

# Save the model to Google Drive
with open('/content/drive/MyDrive/person_detection_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model saved to Google Drive for Arduino deployment")
