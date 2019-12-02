defmodule LayerTest do
  use ExUnit.Case
  doctest Layer
  
  import Matrex
  
  test "dense layer forward propogation" do
    dense_layer = Dense.dense(%{shape_input: {3,3}, n: 3})
    {status, sgd} = SGD.sgd(%{w_: nil, momentum: 0.0, learning_rate: 0.01})
    Dense.init!(dense_layer, sgd)
    Dense.put(dense_layer, :weights, Matrex.new([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]))
    x = Matrex.new([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    updates = Dense.forward_propogate(dense_layer, x)
    
    assert updates == Matrex.new([[0.2, 0.2, 0.2], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
  end
  
  test "dense layer back propogation" do
    dense_layer = Dense.dense(%{shape_input: {1,2}, n: 2})
    {status, sgd} = SGD.sgd(%{w_: nil, momentum: 0.0, learning_rate: 0.01})
    Dense.init!(dense_layer, sgd)
    Dense.put(dense_layer, :weights, Matrex.new([[0.5, 0.5], [0.5, 0.5]]))
    Dense.put(dense_layer, :layer_input, Matrex.new([[1, 0]]))  
    err = Matrex.new([[0.5, 0.5]])
    w = Dense.get(dense_layer, :weights) 
    bias = Dense.get(dense_layer, :bias)
    updates = Dense.backward_propogate(dense_layer, err)
    
    assert w == Matrex.new([[0.5, 0.5], [0.5, 0.5]])
    assert bias == Matrex.new([[0.0, 0.0]])
    assert updates == Matrex.new([[0.5, 0.5]])
  end
   
  test "flatten layer forward propogation" do
    {status, flatten_layer} = Flatten.flatten(%{input_shape: {3, 3}})
    m = Matrex.new([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    v = Flatten.forward_propogate(flatten_layer, m)
    
    assert v == Matrex.new([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
  end
  
  test "flatten layer back propogation" do
  end
   
  test "dropout layer forward propogate" do
    {status, dropout_layer} = Dropout.dropout(%{})
    m = Matrex.ones(10, 10)
    drops = Dropout.forward_propogate(dropout_layer, m)
    sum = Matrex.sum(drops)
    
    # dropout should have converted 20% of the matrix (a hundred ones)
    # to zero and with a default of 0.2 given probabality the sum will
    # ~80 ones
    
    assert 75 < sum and sum < 85
  end
   
  test "dropout layer backward propogate" do
    {status, dropout_layer} = Dropout.dropout(%{})
    m = Matrex.ones(10, 10)
    Dropout.forward_propogate(dropout_layer, m)
    
    back_drops = Dropout.backward_propogate(dropout_layer, m)
    sum = Matrex.sum(back_drops)
    
    # dropout should have converted 20% of the matrix (a hundred ones)
    # to zero and with a default of 0.2 given probabality the sum will
    # ~80 ones
    
    assert 70 < sum and sum < 90
  end
   
  test "batch norm layer forward propogate" do
    {status, sgd} = SGD.sgd(%{w_: nil, momentum: 0.0, learning_rate: 0.01})
    {status, bn_layer} = BatchNormalization.batchnorm(%{})
    BatchNormalization.put(bn_layer, :shape_input, {2,2})
    BatchNormalization.init!(bn_layer, sgd)
    m = Matrex.new([[1, 2], [3, 4]])
    accum_grad = Matrex.new([[0.1, 0.2]])
    
    forward_m = BatchNormalization.forward_propogate(bn_layer, m)
    backward_m = BatchNormalization.backward_propogate(bn_layer, m)
    
    assert forward_m == Matrex.new([[-0.99503719, -0.99503719], [0.99503719, 0.99503719]])
    #assert backward_m == Matrex.new([[0.0, 0.0], [0.0, 0.0]])
  end
end

defmodule ActivationsTest do
  use ExUnit.Case
  doctest Activations
  
  import Matrex

  test "activation propogatation with sigmoid" do
    {status, activ_layer} = Activation.activation(:sigmoid)
    m = Matrex.new([[0.5, 0.5], [0.5, 0.5]])
    
    f_m = Activation.forward_propogate(activ_layer, m)  
    forward_m = Matrex.apply(f_m, fn x -> Float.round(x, 5) end)
    
    b_m = Activation.backward_propogate(activ_layer, Matrex.new([[0.25, 0.25]]))  
    backward_m = Matrex.apply(b_m, fn x -> Float.round(x, 5) end)
    
    assert forward_m == Matrex.new([[0.62246, 0.62246], [0.62246, 0.62246]])
    assert backward_m == Matrex.new([[0.05875, 0.05875], [0.05875, 0.05875]])
  end

  test "activation propogatation with tanh" do
    {status, activ_layer} = Activation.activation(:tanh)
    m = Matrex.new([[0.5, 0.5], [0.5, 0.5]])
    
    f_m = Activation.forward_propogate(activ_layer, m)  
    forward_m = Matrex.apply(f_m, fn x -> Float.round(x, 5) end)
    
    b_m = Activation.backward_propogate(activ_layer, Matrex.new([[0.25, 0.25]]))  
    backward_m = Matrex.apply(b_m, fn x -> Float.round(x, 5) end)
    
    assert forward_m == Matrex.new([[0.46212, 0.46212], [0.46212, 0.46212]])
    assert backward_m == Matrex.new([[0.19661, 0.19661], [0.19661, 0.19661]])
  end

  test "activation propogatation with relu" do
    {status, activ_layer} = Activation.activation(:relu)
    m = Matrex.new([[0.5, 0.5], [0.5, 0.5]])
    
    forward_m = Activation.forward_propogate(activ_layer, m)  
    
    backward_m = Activation.backward_propogate(activ_layer, Matrex.new([[0.25, 0.25]]))  
    
    assert forward_m == Matrex.new([[0.5, 0.5], [0.5, 0.5]])
    assert backward_m == Matrex.new([[0.25, 0.25], [0.25, 0.25]])
  end
end
