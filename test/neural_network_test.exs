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
    
    training_set = [{Matrex.new([[0.0, 0.0]]), Matrex.new([[0.0]])}, 
                    {Matrex.new([[0.0, 1.0]]), Matrex.new([[1.0]])}, 
                    {Matrex.new([[1.0, 0.0]]), Matrex.new([[1.0]])}, 
                    {Matrex.new([[1.0, 1.0]]), Matrex.new([[0.0]])}]

    Enum.reduce(1..5000, [], fn(_, _) -> 
      Enum.reduce(training_set, [], fn({x, y}, _) ->
        outputs = NeuralNetwork.forward_propogate(nn, x)
        output_deltas = Matrex.apply(outputs, fn y_output -> y_output * (1 - y_output) * (y_output - Matrex.at(y, 1, 1)) end)
        NeuralNetwork.backward_propogate(nn, output_deltas)
      end)
    end)
    
    is_within? = fn (y, actual, delta) -> (actual - delta) < y and (actual + delta) > y end
    
    y_0_0 = NeuralNetwork.forward_propogate(nn, Matrex.new([[0.0, 0.0]]))
    y_0_1 = NeuralNetwork.forward_propogate(nn, Matrex.new([[0.0, 1.0]]))
    y_1_0 = NeuralNetwork.forward_propogate(nn, Matrex.new([[1.0, 0.0]]))
    y_1_1 = NeuralNetwork.forward_propogate(nn, Matrex.new([[1.0, 1.0]]))
    
    # tolerance here is generous and indicates that deep learner architecture is not
    # optimal, my guess is that the tanh activation layer shouldn't be used b/c the
    # tanh function is known to have some issues in some cases
    # but now I'm just concern with the DL working correctly not accuracy
    
    tolerance = 0.25
    
    assert is_within?.(Matrex.at(y_0_0, 1, 1), 0, tolerance)
    assert is_within?.(Matrex.at(y_0_1, 1, 1), 1, tolerance)
    assert is_within?.(Matrex.at(y_1_0, 1, 1), 1, tolerance)
    assert is_within?.(Matrex.at(y_1_1, 1, 1), 0, tolerance)
  end
end
