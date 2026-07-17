# * Multi_modal


def cross_attention_block(ecg_input, acc_input):

    # ecg_input = Input(shape=ecg_input_shape)
    ecg_y = Permute((2, 1))(ecg_input)
    ecg_y = Conv1D(128, 8, padding='same',
                   kernel_initializer='he_uniform')(ecg_y)
    ecg_y = BatchNormalization()(ecg_y)
    ecg_y = Activation('relu')(ecg_y)
    ecg_y = squeeze_excite_block(ecg_y)

    ecg_y = Conv1D(256, 5, padding='same',
                   kernel_initializer='he_uniform')(ecg_y)
    ecg_y = BatchNormalization()(ecg_y)
    ecg_y = Activation('relu')(ecg_y)
    ecg_y = squeeze_excite_block(ecg_y)

    ecg_y = Conv1D(128, 3, padding='same',
                   kernel_initializer='he_uniform')(ecg_y)
    ecg_y = BatchNormalization()(ecg_y)
    ecg_y = Activation('relu')(ecg_y)

    ecg_y = GlobalAveragePooling1D()(ecg_y)
    ecg_y = Reshape((1, 128))(ecg_y)

    # acc_input = Input(shape=acc_input_shape)
    acc_y = Permute((2, 1))(acc_input)
    acc_y = Conv1D(128, 8, padding='same',
                   kernel_initializer='he_uniform')(acc_y)
    acc_y = BatchNormalization()(acc_y)
    acc_y = Activation('relu')(acc_y)
    acc_y = squeeze_excite_block(acc_y)

    acc_y = Conv1D(256, 5, padding='same',
                   kernel_initializer='he_uniform')(acc_y)
    acc_y = BatchNormalization()(acc_y)
    acc_y = Activation('relu')(acc_y)
    acc_y = squeeze_excite_block(acc_y)

    acc_y = Conv1D(128, 3, padding='same',
                   kernel_initializer='he_uniform')(acc_y)
    acc_y = BatchNormalization()(acc_y)
    acc_y = Activation('relu')(acc_y)

    acc_y = GlobalAveragePooling1D()(acc_y)
    acc_y = Reshape((1, 128))(acc_y)

    # total_features = Concatenate()([acc_y, ecg_y])
    # total_features = Reshape((1, 128))(total_features)

    cross_attention_output_layer1 = MultiHeadAttention(
        num_heads=6, key_dim=128, dropout=0.2)(query=ecg_y, value=acc_y, key=acc_y)

    cross_attention_output_layer1 = LayerNormalization(
        epsilon=1e-6)(cross_attention_output_layer1 + acc_y)

    cross_attention_output_layer2 = MultiHeadAttention(
        num_heads=6, key_dim=128, dropout=0.2)(query=cross_attention_output_layer1, value=acc_y, key=acc_y)

    cross_attention_output = LayerNormalization(
        epsilon=1e-6)(cross_attention_output_layer2 + cross_attention_output_layer1)

    cross_features = GlobalAveragePooling1D()(cross_attention_output)

    return cross_features


def OS_Block(input_tensor, filters_list, kernel_sizes_list):
    x = input_tensor

    for layer_idx, (filters, kernel_sizes) in enumerate(zip(filters_list, kernel_sizes_list)):
        # Before the channel-scale refinement layer [1, 2],
        # transpose the output of the previous multi-time-scale layers.
        if layer_idx == 2:
            x = Permute((2, 1))(x)

        conv_layers = []
        for kernel_size in kernel_sizes:
            conv = Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding='same',
                kernel_initializer='he_uniform'
            )(x)
            conv = BatchNormalization()(conv)
            conv = ReLU()(conv)
            conv_layers.append(conv)

        if len(conv_layers) > 1:
            x = Concatenate()(conv_layers)
        else:
            x = conv_layers[0]

    return x

# Squeeze-and-Excite Block implementation


def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    filters = input.shape[-1]  # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu',
               kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid',
               kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se

# MLSTM-FCN combined model with OS-Block for ECG data and MLSTM-FCN for accelerometer data


def combined_model(ecg_input_shape, acc_input_shape):
    # ECG data through OS-Block
    # ! ECG数据输入分支
    ecg_input = Input(shape=ecg_input_shape)

    # new_input = Permute((2, 1))(ecg_input)

    filters_list = [64, 128, 256]
    kernel_sizes_list = [
        [3, 5, 7, 11, 43, 73, 97, 127, 197, 307],  # layer 1
        # [3, 5, 7, 11, 17, 23, 31, 43, 73, 97, 127, 197, 307],  # layer 1
        # [3, 5, 7, 11, 43, 97, 127],  # layer 1
        # [3, 5, 7, 11, 17, 23, 73],  # layer 1
        [3, 5, 7, 11, 43, 73, 97, 127, 197, 307],  # layer 2
        # [3, 5, 7, 11, 17, 23, 31, 43, 73, 97, 127, 197, 307],  # layer 2
        # [3, 5, 7, 11, 43, 97, 127],  # layer 2
        # [3, 5, 7, 11, 17, 23, 73],  # layer 2
        [1, 2]
    ]

    ecg_features = OS_Block(ecg_input, filters_list, kernel_sizes_list)
    # Optional: add global average pooling
    ecg_features = GlobalAveragePooling1D()(ecg_features)
    ecg_features = Flatten()(ecg_features)

    # Accelerometer data through MLSTM-FCN-like structure
    # ! Motion数据输入分支
    acc_input = Input(shape=acc_input_shape)

    x = LSTM(8)(acc_input)
    x = Dropout(0.25)(x)

    e = Dense(1, activation='tanh')(x)
    e = Flatten()(e)
    a = Activation('softmax')(e)
    temp = RepeatVector(8)(a)
    temp = Permute([2, 1])(temp)
    x = Multiply()([x, temp])
    x = Lambda(lambda values: tf.compat.v1.keras.backend.sum(values, axis=1))(x)

    y = Permute((2, 1))(acc_input)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    acc_features = Concatenate()([x, y])

    cross_features = cross_attention_block(ecg_input, acc_input)

    # Feature fusion
    combined_features = Concatenate()(
        [ecg_features, acc_features, cross_features])

    # Fully connected and output layer
    x = Dense(64, activation='relu')(combined_features)
    output = Dense(1, activation='sigmoid')(x)

    # Define and compile model
    model = Model(inputs=[ecg_input, acc_input], outputs=output)

    model.summary()

    return model
