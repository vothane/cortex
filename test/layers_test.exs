defmodule LayerTest do
  use ExUnit.Case
  doctest Layer
  
  import Matrex
  
  test "dense layer forward propogation" do
    {status, dense_layer} = Dense.dense(%{shape_input: {3,3}, n: 3})
    {status, sgd} = SGD.sgd(%{w_: nil, momentum: 0.0, learning_rate: 0.01})
    Dense.init(dense_layer, sgd)
    Dense.put(dense_layer, :weights, Matrex.new([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]))
    x = Matrex.new([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    updates = Dense.forward_propogate(dense_layer, x)
    
    assert updates == Matrex.new([[0.2, 0.2, 0.2], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
  end
  
  test "dense layer back propogation" do
    {status, dense_layer} = Dense.dense(%{shape_input: {1,2}, n: 2})
    {status, sgd} = SGD.sgd(%{w_: nil, momentum: 0.0, learning_rate: 0.01})
    Dense.init(dense_layer, sgd)
    Dense.put(dense_layer, :weights, Matrex.new([[0.5, 0.5], [0.5, 0.5]]))
    Dense.put(dense_layer, :layer_input, Matrex.new([[1, 0]]))  
    err = Matrex.new([[0.5, 0.5]])
    w = Dense.get(dense_layer, :weights) 
    bias = Dense.get(dense_layer, :bias)
    updates = Dense.backward_propogate(dense_layer, err)
    
    assert w == Matrex.new([[0.5, 0.5], [0.5, 0.5]])
    #assert bias == Matrex.new([[0.5, 0.5]])
    assert updates == Matrex.new([[0.5, 0.5]])
  end
end

defmodule DenseTest do
  use ExUnit.Case
  doctest Layer

  test "" do
    assert 1 == 1
  end
end
