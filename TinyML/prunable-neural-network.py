import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os

# Check versions (important for compatibility)
print(f"TensorFlow version: {tf.__version__}")
print(f"TensorFlow Model Optimization Toolkit version: {tfmot.__version__}")

# Create a simple dataset for testing
def create_test_data(num_examples=1000, img_height=224, img_width=224):
    x_train = np.random.rand(num_examples, img_height, img_width, 3).astype(np.float32)
    y_train = np.random.randint(0, 10, size=(num_examples,))
    y_train = keras.utils.to_categorical(y_train, 10)
    return x_train, y_train

# Build a model with pre-trained weights that can be pruned
def build_prunable_model():
    # Load a pre-trained model (MobileNetV2 in this case)
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the pre-trained layers
    base_model.trainable = False
    
    # Create a new model on top
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Apply pruning to the model
def apply_pruning(model, train_data, train_labels):
    # Define pruning parameters
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=0,
            end_step=1000
        )
    }
    
    # Apply pruning to the entire model (except the base model)
    pruning_layers = []
    for layer in model.layers:
        # Skip the base model (already trained)
        if isinstance(layer, keras.models.Sequential) or isinstance(layer, keras.models.Model):
            continue
        pruning_layers.append(layer.name)
    
    # Apply pruning to selected layers
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    model_for_pruning = prune_low_magnitude(model, **pruning_params)
    
    # Recompile the model
    model_for_pruning.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Add pruning callback
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=tempfile.mkdtemp()),
    ]
    
    # Train the model with pruning
    batch_size = 32
    model_for_pruning.fit(
        train_data, train_labels,
        batch_size=batch_size,
        epochs=2,
        callbacks=callbacks,
        validation_split=0.1
    )
    
    return model_for_pruning

# Save and convert the model for deployment
def save_and_convert_model(model):
    # Strip the pruning wrappers
    model_for_export = tfmot.sparsity.keras.strip_pruning(model)
    
    # Save the pruned model
    pruned_model_path = os.path.join(tempfile.gettempdir(), "pruned_model")
    tf.keras.models.save_model(
        model_for_export, 
        pruned_model_path, 
        include_optimizer=False
    )
    print(f"Pruned model saved to: {pruned_model_path}")
    
    # Check the size reduction
    model_for_export.summary()
    
    return pruned_model_path

# Main execution
def main():
    # Create test data
    x_train, y_train = create_test_data()
    
    # Build the model with pre-trained weights
    model = build_prunable_model()
    print("Original model summary:")
    model.summary()
    
    # Apply pruning
    pruned_model = apply_pruning(model, x_train, y_train)
    
    # Save the pruned model
    pruned_model_path = save_and_convert_model(pruned_model)
    
    # Compare model sizes
    original_model_size = sum(np.prod(v.get_shape()) for v in model.trainable_variables)
    pruned_model_size = sum(np.prod(v.get_shape()) for v in pruned_model.trainable_variables)
    compression_ratio = (1 - pruned_model_size / original_model_size) * 100
    
    print(f"Original model trainable parameters: {original_model_size}")
    print(f"Pruned model trainable parameters: {pruned_model_size}")
    print(f"Compression ratio: {compression_ratio:.2f}%")
    
    return pruned_model_path

if __name__ == "__main__":
    main()
