defmodule OptimizerTest do
  use ExUnit.Case
  doctest Optimizer
  
  test "sgd" do
    {status, sgd} = SGD.sgd(%{w_: nil, momentum: 0.1, learning_rate: 0.1})
    updated = SGD.update(sgd, Matrex.new([[0.5, 0.5]]), Matrex.new([[0.25, 0.25]]))
    assert updated == Matrex.new([[0.4775, 0.4775]])
  end
end  