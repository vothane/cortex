defmodule LossTest do
  use ExUnit.Case
  doctest Loss
  
  test "square loss" do
    y = Matrex.new([[0.5, 0.5]])
    p = Matrex.new([[0.25, 0.75]])
    ce = %SquareLoss{}
    loss = Loss.loss(ce, y, p)
    grad = Loss.gradient(ce, y, p)
    
    assert loss == Matrex.new(([[0.03125, 0.03125]]))
    assert grad == Matrex.new([[-0.25, 0.25]])
  end
  
  test "cross entropy loss" do
    y = Matrex.new([[0.5, 0.5]])
    p = Matrex.new([[0.25, 0.75]])
    ce = %CrossEntropy{}
    loss = Loss.loss(ce, y, p)
    grad = Loss.gradient(ce, y, p)
    
    #assert loss == Matrex.new([[0.83698822, 0.83698822]])
    #assert grad == Matrex.new([[-1.33333333, 1.33333333]])
  end
end  