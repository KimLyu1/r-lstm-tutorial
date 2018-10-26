# https://blogs.rstudio.com/tensorflow/posts/2017-12-20-time-series-forecasting-with-recurrent-neural-networks/
# Data: "https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip"

setwd("~/Documents/code/r-keras-lstm/")

library(tibble)
library(readr)
library(keras)

# ---- Import and preprocess ----

data <- read_csv("jena_climate_2009_2016.csv")

# The challenge:
# Given data going back `lookback` timesteps, and sapled every `steps` steps,
# can we predict the temperature in `delay` timesteps?

# Pictorially, we are trying to predict the value of our independent variable
# at observation t, using the observed values between `i` - `lookback` and `i`

# |--------------------i---------t--------------> observations
# |------lookback------|--delay--|

# This data has 144 observations per day. We will use `lookback = 1440` to look
# back ten days.
# We will use `steps = 6` to look at one data point per hour
# `delay = 144` - we are predicting one day ahead

# First we need to ensure the data is a numeric matrix. In this case that's easy
# but we will need to discard the first col (date) and convert to a matrix.
data <- data.matrix(data[, -1])

# Extract a training set using the first 75% of the rows.
train_data <- data[1:200000, ]

# Calculate the mean and stdev for each column using `apply()`. Note that we are
# scaling by the mean and sd of the training set.
mean <- apply(train_data, 2, mean)
stdev <- apply(train_data, 2, sd)
data <- scale(data, center = mean, scale = stdev)

# ---- Creating the generic batch generator function ----

# First define a generic generator function:
generator <- function(data, lookback, delay, min_index, max_index,
                      shuffle = FALSE, batch_size = 128, step = 6) {

    # If no max_index has been passed, set the max index to the latest data
    # point we can try and compute with, i.e. the one that is `delay + 1` back
    # from the final observation.
    if (is.null(max_index)) {
        max_index <- nrow(data) - delay - 1
    }

    # The `i` variable will keep track of where we are in the data. Initially
    # it is set to the first observation we can use that will allow us to see
    # `lookback` steps of previous data.
    i <- min_index + lookback

    # The environment setup is now done - the generator now needs to return a
    # function we can use to subset the input array.
    function() {

        # First we need to decide which rows are going to be sampled.
        if (shuffle) {
            # If shuffle is true, sample a random `batch_size` rows between
            # our first viable observation and `max_index`.
            rows <- sample(c((min_index + lookback):max_index),
                           size = batch_size)
        } else {
            # If we're not shuffling we want to choose our next consecutive
            # set of rows.

            # First we need to check if this batch will take us past our
            # `max_index`. If so, we reset i (via superassignment) to the
            # earliest viable observation (which is `lookback` observations
            # after `min_index`).
            if (i + batch_size >= max_index) {
                i <<- min_index + lookback
            }

            # The `rows` variable contains the indices of the rows to that we
            # will predict for in the batch. It starts at i, and ends either
            # after `batch_size` rows or at `max_index`.
            rows <- c(i:min(i + batch_size - 1, max_index))

            # Then we increment i (via superassignment) to keep track of where
            # we are in the data.
            i <<- i + length(rows)
        }

        # Now that we know which rows are going to be included in the batch we
        # need to get them into the right format to feed into the LSTM.

        # We create `samples`, an array of zeroes of the correct dimensions to
        # feed into the LSTM. The correct shape is:
        #
        #    [num_samples, num_timesteps, num_features]
        #
        # More info: https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
        samples <- array(0,
                         dim = c(length(rows),      # Number of samples
                                 lookback / step,   # Number of timesteps # TODO: What happens if this divison results in a non-integer? Do we want %/% (integer division) instead?
                                 dim(data)[[-1]]))  # Number of features

        # We alo create a `targets` array. This will hold the values we are
        # trying to predict contains one target value for each input row.
        targets <- array(0, dim = c(length(rows)))

        # Now that the arrays are set up we need to populate them with the
        # relevant data. This loop identifies the indices that contain the
        # data to be included in the batch, and copies the relevant values to
        # our output arrays.
        for (j in 1:length(rows)) {
            # Determine the indices of the rows we need to extract data from.
            indices <- seq(rows[[j]] - lookback,            # The first observation of our input data to be used to predict for the current observation
                           rows[[j]] - 1,                   # The last observation to be used as a predictor (i.e. the one immediately before the one we're trying to predict)
                           length.out = dim(samples)[[2]])  # The number of timesteps to be sampled. # TODO: What if this results in non-integer index values?

            # Move each set of variables (i.e. each row in the input data)
            # into the corresponding slot in the output array.
            # Each iteration through the loop adds a new sample of dimensions
            #
            #    [num_timesteps, num_features]
            #
            # into the output, maintaining our overall aim of
            #
            #    [num_samples, num_timesteps, num_features]
            #
            # as our output shape.
            samples[j, , ] <- data[indices, ]

            # Now we need to populate the target array in the same way. We do
            # this by inserting the target value from the j-th + `delay` row,
            # i.e. the row `delay` steps ahead and therefore our target ouput
            # value.
            targets[[j]] <- data[rows[[j]] + delay, 2]  # TODO: The 2 here hard-codes the target as column 2 of the input data.
        }

        # All done! We can now return a list containing our samples and targets

        return(list(samples, targets))
    }
}

# ---- Creating specific batch generators ----

# Now that the generic function is in place we can use it to create functions
# to feed data in for training, validation and testing. The only differences
# between them are their `max_` and `min_index` values (i.e. which section of
# the data they cover) and whether or not they shuffle their output.

# TODO: Generalise to a single function that will return all three, probably
#       in a list. Should calculate train, test and validation min and max
#       indices, and accept target variable as a parameter to address the
#       hard-coding issue above.

# These are the parameters to be used in the analysis:
lookback <- 1440
step <- 6
delay <- 144
batch_size <- 128

# Training set generator
train_gen <- generator(
    data = data,
    lookback = lookback,
    delay = delay,
    min_index = 1,
    max_index = 200000,
    shuffle = TRUE,  # TODO: why is this OK for training?
    step = step,
    batch_size = batch_size
)

# Validation set generator
val_gen <- generator(
    data = data,
    lookback = lookback,
    delay = delay,
    min_index = 200001,
    max_index = 300000,
    step = step,
    batch_size = batch_size
)

# Test set generator
test_gen <- generator(
    data = data,
    lookback = lookback,
    delay = delay,
    min_index = 300001,
    max_index = NULL,
    step = step,
    batch_size = batch_size
)

# How many steps should we draw from val_gen and test_gen in order to see the
# entire validation and test sets?
# TODO: Wrap this up in a meta-function to remove the need for hard-coding
val_steps <- (300000 - 200001 - lookback) / batch_size
test_steps <- (nrow(data) - 300001 - lookback) / batch_size

# ---- Basic non-machine learning approach ----

# In the case of the demo dataset a naive approach is to assume that the temp
# in 24 hours' time will be the same as it is right now. The code below uses
# MAE (mean absolute error) as a measure of accuracy.
evaluate_naive_method <- function() {

    # Empty vector to hold MAE of each batch
    batch_maes <- c()

    for (step in 1:val_steps) {
        c(samples, targets) %<-% val_gen()  # Note the use of %<-% for multiple assignment. See https://cran.r-project.org/web/packages/zeallot/index.html
        # For the preicted values we will just take the values from the samples
        # array and assume that the value in x steps time will be the same
        preds <- samples[, dim(samples)[[2]], 2]

        # Calculate mean absolute error of the "predicted" and actual values
        # TODO: Note the hard-coded 2 here (the second one - selecting the
        #       second variable). It will need amending if this approach is used
        #       on different data. Again, it would be nice to wrap this up into
        #       a proper function.
        mae <- mean(abs(preds - targets))
        batch_maes <- c(batch_maes, mae)
    }

    return(mean(batch_maes))

}

# Run the naive estimator to get a baseline error
naive_mae <- evaluate_naive_method()
# Convert back to source units by multiplying by stdev and adding mean
# TODO: Again this has a hard-coded 2 - this will need amending
naive_error <- (naive_mae * stdev[[2]]) + mean[[2]]


# ---- Basic maching learning method ----

# Before running an expensive LSTM model, let's try a quick and dirty
# fully-connected neural net.
model <- keras_model_sequential() %>%
    # As this is the first layer in the model we are required to specify the
    # shape of the input data. Its shape is
    #
    #    [num_timesteps, num_features]
    #
    layer_flatten(input_shape = c(lookback / step,
                                  dim(data)[-1])) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 1)

# Compile the model
model %>% compile(
    optimizer = optimizer_rmsprop(),
    loss = "mae"
)

# Train the model and store the output in `history`
history <- model %>% fit_generator(
    generator = train_gen,
    steps_per_epoch = 500,        # Tweak
    epochs = 10,                  # Tweak
    validation_data = val_gen,
    validation_steps = val_steps
)

# Show the loss as training progressed
print(history)
plot(history)

# The loss here is not much different from the naive approach

# ---- A first recurrent baseline ----

model <- keras_model_sequential() %>%
    layer_gru(units = 32, input_shape = list(NULL, dim(data)[[-1]])) %>%
    layer_dense(units = 1)

model %>% compile(
    optimizer = optimizer_rmsprop(),
    loss = "mae"
)

history <- model %>% fit_generator(
    train_gen,
    steps_per_epoch = 500,
    epochs = 20,
    validation_data = val_gen,
    validation_steps = val_steps
)

# ---- RNN with dropout ----

model <- keras_model_sequential() %>%
    layer_gru(units = 32, dropout = 0.2, recurrent_dropout = 0.2,
              input_shape = list(NULL, dim(data)[[-1]])) %>%
    layer_dense(units = 1)

model %>% compile(
    optimizer = optimizer_rmsprop(),
    loss = "mae"
)

history <- model %>% fit_generator(
    train_gen,
    steps_per_epoch = 500,
    epochs = 40,
    validation_data = val_gen,
    validation_steps = val_steps
)

# ---- Stacking recurrent layers ----

model <- keras_model_sequential() %>%
    layer_gru(units = 32,
              dropout = 0.1,
              recurrent_dropout = 0.5,
              return_sequences = TRUE,
              input_shape = list(NULL, dim(data)[[-1]])) %>%
    layer_gru(units = 64, activation = "relu",
              dropout = 0.1,
              recurrent_dropout = 0.5) %>%
    layer_dense(units = 1)

model %>% compile(
    optimizer = optimizer_rmsprop(),
    loss = "mae"
)

history <- model %>% fit_generator(
    train_gen,
    steps_per_epoch = 500,
    epochs = 40,
    validation_data = val_gen,
    validation_steps = val_steps
)

# ---- Bidirectional RNNs ----

# This is for text processing rather than TS forecasting

# Number of words to consider as features
max_features <- 10000

# Cuts off texts after this number of words
maxlen <- 500

imdb <- dataset_imdb(num_words = max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb

# Reverses sequences
x_train <- lapply(x_train, rev)
x_test <- lapply(x_test, rev)

# Pads sequences
x_train <- pad_sequences(x_train, maxlen = maxlen)  <4>
    x_test <- pad_sequences(x_test, maxlen = maxlen)

model <- keras_model_sequential() %>%
    layer_embedding(input_dim = max_features, output_dim = 128) %>%
    layer_lstm(units = 32) %>%
    layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("acc")
)

history <- model %>% fit(
    x_train, y_train,
    epochs = 10,
    batch_size = 128,
    validation_split = 0.2
)

# Bidirectional RNN

model <- keras_model_sequential() %>%
    layer_embedding(input_dim = max_features, output_dim = 32) %>%
    bidirectional(
        layer_lstm(units = 32)
    ) %>%
    layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("acc")
)

history <- model %>% fit(
    x_train, y_train,
    epochs = 10,
    batch_size = 128,
    validation_split = 0.2
)

# Run bidirectional RNN on temperature task

model <- keras_model_sequential() %>%
    bidirectional(
        layer_gru(units = 32), input_shape = list(NULL, dim(data)[[-1]])
    ) %>%
    layer_dense(units = 1)

model %>% compile(
    optimizer = optimizer_rmsprop(),
    loss = "mae"
)

history <- model %>% fit_generator(
    train_gen,
    steps_per_epoch = 500,
    epochs = 40,
    validation_data = val_gen,
    validation_steps = val_steps
)
