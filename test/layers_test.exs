defmodule LayerTest do
  use ExUnit.Case
  doctest Layer
  
  import Matrex
  
  test "............" do
    dense_layer = Dense.dense(%{shape_input: {5,5}, n: 4})
    IO.inspect(dense_layer)
    assert dense_layer == 1
  end
end

defmodule DenseTest do
  use ExUnit.Case
  doctest Layer

  test "............" do
    assert 1 == 1
  end
end
