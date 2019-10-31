defmodule NeuralNetworkTest do
  use ExUnit.Case
  doctest NeuralNetwork
  
  import Matrex
  import NeuralNetwork
  import SGD
  import Activation
  import TanH
  import Sigmoid
  
  test "cortex with XOR problem without loss function" do
    {status, sgd} = sgd(%{w_: nil, momentum: 0.0, learning_rate: 0.1})
    {status, nn} = neural_network(sgd)
    {status, tanh_layer} = activation(:tanh)
    {status, sigmoid_layer} = activation(:sigmoid)
    
    NeuralNetwork.add(nn, Dense.dense(%{shape_input: {1,2}, n: 2}))   
    NeuralNetwork.add(nn, tanh_layer)
    NeuralNetwork.add(nn, Dense.dense(%{n: 1}))
    NeuralNetwork.add(nn, sigmoid_layer)
    
    NeuralNetwork.forward_propogate(nn, Matrex.new([[1, 0]]))
  end

end
