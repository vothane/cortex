defmodule NeuralNetworkTest do
  use ExUnit.Case
  doctest NeuralNetwork
  
  test "cortex with XOR problem without loss function" do
    {status, sgd} = SGD.sgd(%{w_: nil, momentum: 0.0, learning_rate: 0.01})
    {status, nn} = NeuralNetwork.neural_network(sgd)
    NeuralNetwork.add(nn, Dense.dense(%{shape_input: {1,2}, n: 52}))
    {status, activ_layer} = Activation.activation(%{activation_fn: %Sigmoid{}})
    NeuralNetwork.add(nn, activ_layer)
    NeuralNetwork.add(nn, Dense.dense(%{n: 8}))
  end

end
