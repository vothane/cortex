defmodule OptimizerTest do
  use ExUnit.Case
  doctest Optimizer
  
  test "sgd" do
    {status, sgd} = SGD.sgd(%{w_: nil, momentum: 0.1, learning_rate: 0.1})
    updated = SGD.update!(sgd, Matrex.new([[0.5, 0.5]]), Matrex.new([[0.25, 0.25]]))
    assert updated == Matrex.new([[0.4775, 0.4775]])
  end

  test "RMSprop" do
    {status, rmsp} = RMSprop.rmsp(%{})
    weights = Matrex.new([[ 0.2567016, -0.28198047], [-0.21376074, 0.37581967]])
    gradients = Matrex.new([[50.1850741, -50.1850741 ], [-296.85395415, 296.85395415]])
    updated = Matrex.apply(RMSprop.update!(rmsp, weights, gradients), fn x -> Float.round(x, 5) end)
    assert updated == Matrex.new([[0.22508, -0.25036], [-0.18214, 0.34420]])
  end
end  