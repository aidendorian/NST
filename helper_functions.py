import tensorflow as tf

def content_loss(generated_features, content_features):
    """
    Calculates the content loss between the content image and generated image.
    
    Args:
        generated_features: Feature map of the generated image.
        content_features: Feature map of the content image.
        
    Returns:
        Content loss (scalar).
    """
    
    return tf.reduce_mean(tf.square(generated_features - content_features))

def gram_matrix(feature_map):
    """
    Calculates the Gram Matrix of the feature map to capture style correlations.
    
    Args:
        feature_map: Feature map of the image (e.g., from VGG19 layers).
        
    Returns:
        Gram Matrix of the feature map, shape [B, C, C].
    """
    
    feature_map = tf.cast(feature_map, tf.float32)
    B, H, W, C = tf.unstack(tf.shape(feature_map))
    x = tf.reshape(feature_map, [B, H * W, C])
    gram = tf.matmul(x, x, transpose_a=True)
    return gram / tf.cast(H * W, tf.float32)

def style_loss(generated_features, style_features, style_weights=0.2):
    """
    Calculates style loss between style image and generated image by comparing Gram matrices.
    
    Args:
        generated_features: List of feature maps of the generated image.
        style_features: List of feature maps of the style image.
        style_weights: Weight for each layer (scalar or list of scalars, default 0.2 for all layers).
        
    Returns:
        Style loss (scalar).
    """
    
    if isinstance(style_weights, (int, float)):
        style_weights = [style_weights] * len(style_features)
    elif len(style_weights) != len(style_features):
        raise ValueError("style_weights must be a scalar or a list matching the number of style features")

    loss = tf.constant(0., dtype=tf.float32)
    for i in range(len(style_features)):
        gram_g = gram_matrix(generated_features[i])
        gram_s = gram_matrix(style_features[i])
        layer_loss = tf.reduce_mean(tf.square(gram_g - gram_s))
        loss += layer_loss * style_weights[i]
    return loss

def total_variation(feature_map):
    """
    Calculates the total variation of the feature map to encourage smoothness.

    Args:
        feature_map: Feature map of the image (e.g., generated image).
        
    Returns:
        Total variation loss (scalar).
    """
    
    return tf.image.total_variation(feature_map)

def total_loss(generated_content, content_features, generated_style, style_features, generated_image, 
               c_weight=1.0, s_weight=1.0, tv_weight=0.01):
    """
    Calculates the total loss, combining content loss, style loss, and total variation loss.

    Args:
        generated_content: Content feature of the generated image.
        content_features: Content feature of the content image.
        generated_style: List of style features of the generated image.
        style_features: List of style features of the style image.
        generated_image: Generated image with the content and style features.
        c_weight: Content loss weight (default 1.0).
        s_weight: Style loss weight (default 1.0).
        tv_weight: Total variation weight (default 0.01 to avoid over-smoothing).
        
    Returns:
        Total loss (scalar).
    """
    
    c_loss = content_loss(generated_content, content_features)
    s_loss = style_loss(generated_style, style_features)
    tv = total_variation(generated_image)
    
    loss = c_loss * c_weight + s_loss * s_weight + tv * tv_weight
    return loss