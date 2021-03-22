defmodule LossTest do
  use ExUnit.Case
  doctest Loss

  import CrossEntropy
  import SquareLoss

  test "square loss" do
    y = Nx.tensor([[0.5, 0.5]])
    p = Nx.tensor([[0.25, 0.75]])
    sqrl = square_loss(%{})
    loss = Loss.loss(sqrl, y, p)
    grad = Loss.gradient(sqrl, y, p)
    
    assert loss == Nx.tensor(([[0.03125, 0.03125]]))
    assert grad == Nx.tensor([[-0.25, 0.25]])
  end
  
  test "cross entropy loss" do
    y = Nx.tensor([[1.0, 0.0, 0.0]])
    p = Nx.tensor([[0.34002627, 0.33873541, 0.32123832]])
    {_, ce} = cross_entropy(%{})
    loss = Loss.loss(ce, y, p)
    grad = Loss.gradient(ce, y, p)
    
    IO.inspect(loss)
    IO.inspect(grad)
  end
  
  test "cross entropy gradient" do
    {_, ce} = cross_entropy(%{})
    assert Matrex.apply(Loss.gradient(ce, Nx.tensor([[0.0, 1.0]]), Nx.tensor([[0.1, 0.9]])), fn x -> Float.round(x, 5) end)
        == Nx.tensor([[1.11111, -1.11111]])
    assert Matrex.apply(Loss.gradient(ce, Nx.tensor([[0.0, 1.0]]), Nx.tensor([[0.2, 0.8]])), fn x -> Float.round(x, 5) end)
        == Nx.tensor([[1.25, -1.25]])
    assert Matrex.apply(Loss.gradient(ce, Nx.tensor([[0.0, 1.0]]), Nx.tensor([[0.3, 0.7]])), fn x -> Float.round(x, 5) end)
        == Nx.tensor([[1.42857, -1.42857]])
    assert Matrex.apply(Loss.gradient(ce, Nx.tensor([[0.0, 1.0]]), Nx.tensor([[0.4, 0.6]])), fn x -> Float.round(x, 5) end)
        == Nx.tensor([[1.66667, -1.66667]])
    assert Matrex.apply(Loss.gradient(ce, Nx.tensor([[0.0, 1.0]]), Nx.tensor([[0.5, 0.5]])), fn x -> Float.round(x, 5) end)
        == Nx.tensor([[2.0, -2.0]])
  end

end