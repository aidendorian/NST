import tensorflow as tf

def content_loss(generated_features, content_features):
    """
    Calculates the content loss between the content image and generated image.
    
    Args:
        generated_features: Feature map of the generated image.
        content_features: Feature map of the content image.
        
    Returns:
        Content loss
    """
    
    
    return tf.reduce_mean(tf.square(generated_features - content_features))

def gram_matrix(feature_map):
    """
    Calculates the Gram Matrix of the feature map.
    
    Args:
        feature map: Feature of the image.
        
    Returns:
        Gram Matrix of the feature map
    """
    tf.cast(feature_map, tf.float32)
    B, H, W, C = tf.unstack(tf.shape(feature_map))
    x = tf.reshape(feature_map, [B, H*W, C])
    gram = tf.matmul(x, x, transpose_a=True)
    return gram / tf.cast(H*W, tf.float32)

def style_loss(generated_features, style_features, style_weights=0.2):
    """
    Calculates style loss between style image and generated image.
    
    Args:
        generated_features: Feature map of the generated image.
        style_features: Feature map of the style image.
        style_weights: Weight of each layer (set to 0.2 as default for all layers)
        
    Return:
        Style Loss
    """
    
    loss = 0.
    for i in range(len(style_features)):
        gram_g = gram_matrix(generated_features[i])
        gram_s = gram_matrix(style_features[i])
        layer_loss = tf.reduce_mean(tf.square(gram_g - gram_s))
        loss += layer_loss*style_weights
    return loss

def total_variation(feature_map):
    """
    Calculates the total variation of the feature_map.

    Args:
        feature_map: Feature map of the image.
        
    Returns:
        Total Variation Loss
    """
    
    return tf.image.total_variation(feature_map)

def total_loss(generated_content, content_features, generated_style, style_features, generated_image, c_weight, s_weight, tv_weight):
    """
    Calculates the total loss, taking into account - content loss, style loss and total variation.

    Args:
        generated_content: Content feature of the generated image.
        content_features: Content feature of the content image.
        generated_style: Style feature of the generated image.
        style_features: Style feature of the style image.
        generated_image: Generated image with the content_features and style_features.
        c_weight: Content loss weight.
        s_weight: Style loss weight.
        tv_weight: Total Variation weight.
        
    Returns:
        Total Loss.
    """
    
    c_loss = content_loss(generated_content, content_features)
    s_loss = style_loss(generated_style, style_features)
    tv = total_variation(generated_image)
    
    loss = c_loss*c_weight + s_loss*s_weight + tv*tv_weight
    return loss