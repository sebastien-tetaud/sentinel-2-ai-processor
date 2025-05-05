import segmentation_models_pytorch as smp

def define_model(
    name,
    encoder_name,
    out_channels=3,
    in_channel=3,
    encoder_weights=None,
    activation=None,

):
    # Get the model class dynamically based on name
    try:
        # Get the model class from segmentation_models_pytorch
        ModelClass = getattr(smp, name)


        # Create the model
        model = ModelClass(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channel,
            classes=out_channels,
            activation=activation,

        )

        return model


    except AttributeError:
        # If the model name is not found in the library
        raise ValueError(f"Model '{name}' not found in segmentation_models_pytorch. Available models: {dir(smp)}")
