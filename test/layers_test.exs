defmodule LayerTest do
  use ExUnit.Case
  doctest Layer
  
  import Matrex
  
  test "dense layer back propogation" do
    {status, dense_layer} = Dense.dense(%{shape_input: {1,2}, n: 2})
    Agent.update(dense_layer, fn state -> Map.put(state, :weights, Matrex.new([[0.5, 0.5]])) end)
    Agent.update(dense_layer, fn state -> Map.put(state, :input, Matrex.new([[1, 0]])) end)
    updates = Dense.backward_propogate(dense_layer, Matrex.new([[0.5, 0.5]])) 
    W = Agent.get(dense_layer, fn state -> Map.get(state, :weights) end)
    assert W == Matrex.new([[0.5, 0.5]])
    assert updates == Matrex.new([[0.5]])
  end
end

defmodule DenseTest do
  use ExUnit.Case
  doctest Layer

  test "" do
    assert 1 == 1
  end
end
