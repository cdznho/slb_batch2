from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras import layers, models, optimizers, regularizers
def load_base_model():

    model = ResNet50V2(weights=None, input_shape=(256, 256, 3), include_top=False, input_tensor=None, pooling=None, classifier_activation="softmax")
    
    return model

def set_nontrainable_layers(model):
    # Set the first layers to be untrainable
    model.trainable = True
    return model

def add_last_layers(model):
    #reg = regularizers.l1_l2(l1=0.0001, l2=0.0001)

    base_model = set_nontrainable_layers(model)
    flatten_layer = layers.Flatten()
    dense_layer = layers.Dense(410, activation='relu')
    drop_layer = layers.Dropout(0.4)
    prediction_layer = layers.Dense(24, activation='softmax')
    
    model = models.Sequential([
        base_model,
        flatten_layer,
        dense_layer,
        drop_layer,
        prediction_layer
    ])

    return model

def build_model():
  
    model = load_base_model()
    model = add_last_layers(model)
    
    opt = optimizers.Adam(learning_rate=3e-4)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model
