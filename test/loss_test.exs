defmodule LossTest do
  use ExUnit.Case
  doctest Loss

  import CrossEntropy
  import SquareLoss

  test "square loss" do
    y = Matrex.new([[0.5, 0.5]])
    p = Matrex.new([[0.25, 0.75]])
    sqrl = square_loss(%{})
    loss = Loss.loss(sqrl, y, p)
    grad = Loss.gradient(sqrl, y, p)
    
    assert loss == Matrex.new(([[0.03125, 0.03125]]))
    assert grad == Matrex.new([[-0.25, 0.25]])
  end
  
  test "cross entropy loss" do
    {_, ce} = cross_entropy(%{})
    assert Matrex.apply(Loss.loss(ce, Matrex.new([[0.0, 1.0]]), Matrex.new([[0.1, 0.9]])), fn x -> Float.round(x, 5) end)
        == Matrex.new([[0.10536, 0.10536]])
    assert Matrex.apply(Loss.loss(ce, Matrex.new([[0.0, 1.0]]), Matrex.new([[0.2, 0.8]])), fn x -> Float.round(x, 5) end)
        == Matrex.new([[0.22314, 0.22314]])
    assert Matrex.apply(Loss.loss(ce, Matrex.new([[0.0, 1.0]]), Matrex.new([[0.3, 0.7]])), fn x -> Float.round(x, 5) end)
        == Matrex.new([[0.35667, 0.35667]])
    assert Matrex.apply(Loss.loss(ce, Matrex.new([[0.0, 1.0]]), Matrex.new([[0.4, 0.6]])), fn x -> Float.round(x, 4) end)
        == Matrex.new([[0.5108, 0.5108]])
    assert Matrex.apply(Loss.loss(ce, Matrex.new([[0.0, 1.0]]), Matrex.new([[0.5, 0.5]])), fn x -> Float.round(x, 5) end)
        == Matrex.new([[0.69315, 0.69315]])

  end
  
  test "cross entropy gradient" do
    {_, ce} = cross_entropy(%{})
    assert Matrex.apply(Loss.gradient(ce, Matrex.new([[0.0, 1.0]]), Matrex.new([[0.1, 0.9]])), fn x -> Float.round(x, 5) end)
           == Matrex.new([[1.11111, -1.11111]])
    assert Matrex.apply(Loss.gradient(ce, Matrex.new([[0.0, 1.0]]), Matrex.new([[0.2, 0.8]])), fn x -> Float.round(x, 5) end)
           == Matrex.new([[1.25, -1.25]])
    assert Matrex.apply(Loss.gradient(ce, Matrex.new([[0.0, 1.0]]), Matrex.new([[0.3, 0.7]])), fn x -> Float.round(x, 5) end)
           == Matrex.new([[1.42857, -1.42857]])
    assert Matrex.apply(Loss.gradient(ce, Matrex.new([[0.0, 1.0]]), Matrex.new([[0.4, 0.6]])), fn x -> Float.round(x, 5) end)
           == Matrex.new([[1.66667, -1.66667]])
    assert Matrex.apply(Loss.gradient(ce, Matrex.new([[0.0, 1.0]]), Matrex.new([[0.5, 0.5]])), fn x -> Float.round(x, 5) end)
           == Matrex.new([[2.0, -2.0]])
  end

end