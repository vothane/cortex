defmodule LayerTest do
  use ExUnit.Case
  doctest Layer
  
  import Nx
  import Utils

  test "dense layer forward propogation" do
    dense_layer = Dense.dense(%{shape_input: {3,3}, n: 3})
    {status, sgd} = SGD.sgd(%{w_: nil, momentum: 0.0, learning_rate: 0.01})
    Dense.init!(dense_layer, sgd)
    Dense.put(dense_layer, :weights, Nx.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]))
    x = Nx.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    updates = Dense.forward_propogate(dense_layer, x)
    
    assert updates == Nx.tensor([[0.2, 0.2, 0.2], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
  end
  
  test "dense layer back propogation" do
    dense_layer = Dense.dense(%{shape_input: {1,2}, n: 2})
    {status, sgd} = SGD.sgd(%{w_: nil, momentum: 0.0, learning_rate: 0.01})
    Dense.init!(dense_layer, sgd)
    Dense.put(dense_layer, :weights, Nx.tensor([[0.5, 0.5], [0.5, 0.5]]))
    Dense.put(dense_layer, :layer_input, Nx.tensor([[1, 0]]))  
    err = Nx.tensor([[0.5, 0.5]])
    w = Dense.get(dense_layer, :weights) 
    bias = Dense.get(dense_layer, :bias)
    updates = Dense.backward_propogate(dense_layer, err)
    
    assert w == Nx.tensor([[0.5, 0.5], [0.5, 0.5]])
    assert bias == Nx.tensor([[0.0, 0.0]])
    assert updates == Nx.tensor([[0.5, 0.5]])
  end
   
  test "flatten layer forward propogation" do
    {status, flatten_layer} = Flatten.flatten(%{input_shape: {3, 3}})
    m = Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    v = Flatten.forward_propogate(flatten_layer, m)
    
    assert v == Nx.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
  end
  
  test "flatten layer back propogation" do
  end
   
  test "dropout layer forward propogate" do
    {status, dropout_layer} = Dropout.dropout(%{})
    t = new_tensor({10, 10}, fn -> 1 end)
    drops = Dropout.forward_propogate(dropout_layer, t)
    sum = Nx.sum(drops)
    
    # dropout should have converted 20% of the matrix (a hundred ones)
    # to zero and with a default of 0.2 given probabality the sum will
    # ~80 ones
    IO.inspect(Nx.broadcast(sum))
    assert 75 < sum and sum < 85
  end
   
  # test "dropout layer backward propogate" do
  #   {status, dropout_layer} = Dropout.dropout(%{})
  #   m = Utils.new_tensor({10, 10}, fn _ -> 10 end)
  #   Dropout.forward_propogate(dropout_layer, m)
    
  #   back_drops = Dropout.backward_propogate(dropout_layer, m)
  #   sum = Nx.sum(back_drops)
    
  #   # dropout should have converted 20% of the matrix (a hundred ones)
  #   # to zero and with a default of 0.2 given probabality the sum will
  #   # ~80 ones
    
  #   assert 70 < sum and sum < 90
  # end
   
  # test "batch norm layer forward propogate" do
  #   {status, sgd} = SGD.sgd(%{w_: nil, momentum: 0.0, learning_rate: 0.01})
  #   {status, bn_layer} = BatchNormalization.batchnorm(%{})
  #   BatchNormalization.put(bn_layer, :shape_input, {2,2})
  #   BatchNormalization.init!(bn_layer, sgd)
  #   m = Matrex.new([[1, 2], [3, 4]])
  #   accum_grad = Matrex.new([[0.1, 0.2]])
    
  #   forward_m = BatchNormalization.forward_propogate(bn_layer, m)
  #   backward_m = BatchNormalization.backward_propogate(bn_layer, accum_grad)
    
  #   assert forward_m == Matrex.new([[-0.99503719, -0.99503719], [0.99503719, 0.99503719]])
  #   assert backward_m == Matrex.new([[0.0, 0.0], [0.0, 0.0]])
  # end
end

defmodule ActivationsTest do
  use ExUnit.Case
  doctest Activations
  
  import Nx

  test "activation propogatation with sigmoid" do
    {status, activ_layer} = Activation.activation(:sigmoid)
    m = Nx.tensor([[0.5, 0.5], [0.5, 0.5]])
    
    f_m = Activation.forward_propogate(activ_layer, m)  
    forward_m = Nx.map(f_m, fn x -> Float.round(x, 5) end)
    
    b_m = Activation.backward_propogate(activ_layer, Nx.tensor([[0.25, 0.25]]))  
    backward_m = Nx.map(b_m, fn x -> Float.round(x, 5) end)
    
    assert forward_m == Nx.tensor([[0.62246, 0.62246], [0.62246, 0.62246]])
    assert backward_m == Nx.tensor([[0.05875, 0.05875], [0.05875, 0.05875]])
  end

  test "activation propogatation with tanh" do
    {status, activ_layer} = Activation.activation(:tanh)
    m = Nx.tensor([[0.5, 0.5], [0.5, 0.5]])
    
    f_m = Activation.forward_propogate(activ_layer, m)  
    forward_m = Nx.map(f_m, fn x -> Float.round(x, 5) end)
    
    b_m = Activation.backward_propogate(activ_layer, Nx.tensor([[0.25, 0.25]]))  
    backward_m = Nx.map(b_m, fn x -> Float.round(x, 5) end)
    
    assert forward_m == Nx.tensor([[0.46212, 0.46212], [0.46212, 0.46212]])
    assert backward_m == Nx.tensor([[0.19661, 0.19661], [0.19661, 0.19661]])
  end

  test "activation propogatation with relu" do
    {status, activ_layer} = Activation.activation(:relu)
    m = Nx.tensor([[0.5, 0.5], [0.5, 0.5]])
    
    forward_m = Activation.forward_propogate(activ_layer, m)  
    
    backward_m = Activation.backward_propogate(activ_layer, Nx.tensor([[0.25, 0.25]]))  
    
    assert forward_m == Nx.tensor([[0.5, 0.5], [0.5, 0.5]])
    assert backward_m == Nx.tensor([[0.25, 0.25], [0.25, 0.25]])
  end
end
