defmodule NeuralNetworkTest do
  use ExUnit.Case
  doctest NeuralNetwork

  test "vanilla nn" do
    """
    nn = NeuralNetwork(...)
    add(nn, dense(input_shape: {n_features, :any}, n: 64))
    add(nn, activation(:relu))
    add(nn, dense(n: 64))
    add(nn, activation(:relu))
    add(nn, dense(n: 2))   
    add(nn, activation(:softmax))
    
    nn.fit(X)
    
    nn.predict(x)
    """
  end

  test "" do
    """
    cnn = NeuralNetwork(...)
    
    add(cnn, conv2D(n_filters: 16, filter_shape: {3,3}, stride: 1, input_shape: {1,8,8}, padding: :same))
    add(cnn, activation(:relu))
    add(cnn, dropout(0.25))
    add(cnn, batchNormalization())
    add(cnn, conv2D(n_filters: 32, filter_shape: {3,3}, stride: 1, padding: :same))
    add(cnn, activation(:relu))
    add(cnn, dropout(0.25))
    add(cnn, batchNormalization())
    add(cnn, flatten())
    add(cnn, dense(n: 256))
    add(cnn, activation(:relu))
    add(cnn, dropout(0.4))
    add(cnn, batchNormalization())
    add(cnn, dense(n: 10))
    add(cnn, activation(:softmax))
    
    nn.fit(X)
    
    nn.predict(x)
    """
  end
  
  
end
