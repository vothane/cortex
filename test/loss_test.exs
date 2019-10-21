defmodule LossTest do
  use ExUnit.Case
  doctest Loss
  
  test "square loss" do
    y = Matrex.new([[0.5, 0.5]])
    p = Matrex.new([[0.25, 0.75]])
    sqrl = %SquareLoss{}
    loss = Loss.loss(sqrl, y, p)
    grad = Loss.gradient(sqrl, y, p)
    
    assert loss == Matrex.new(([[0.03125, 0.03125]]))
    assert grad == Matrex.new([[-0.25, 0.25]])
  end
  
  test "cross entropy loss" do
    y = Matrex.new([[0.5, 0.5]])
    p = Matrex.new([[0.25, 0.75]])
    ce = %CrossEntropy{}
    loss = Loss.loss(ce, y, p)
    grad = Matrex.apply(Loss.gradient(ce, y, p), fn x -> Float.round(x, 5) end)
    
    assert loss == Matrex.new([[0.83698822, 0.83698822]])
    assert grad == Matrex.new([[-1.33333, 1.33333]])
  end
end  