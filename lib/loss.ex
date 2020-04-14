defmodule Loss do
  @callback loss!(struct, any, any) :: any
  @callback gradient!(struct, any, any) :: any
  
  def loss(loss, actuals, predictions) do
    %module{} = Agent.get(loss, &(&1))
    module.loss!(actuals, predictions)   
  end
  
  def gradient(loss, actuals, predictions) do
    %module{} = Agent.get(loss, &(&1))
    module.gradient!(actuals, predictions)   
  end
end

defmodule SquareLoss do
  defstruct [name: :square_loss]
  
  @behaviour Loss
  
  @impl Loss
  def loss!(y, y_pred), do: Matrex.multiply(0.5, Matrex.apply(Matrex.subtract(y, y_pred), fn x -> :math.pow(x, 2) end))
  
  @impl Loss
  def gradient!(y, y_pred), do: Matrex.multiply(-1, (Matrex.subtract(y, y_pred)))

  def square_loss(%{}) do
    Agent.start_link(fn -> %SquareLoss{} end)
  end
end

defmodule CrossEntropy do
  defstruct [name: :cross_entropy]
  
  @behaviour Loss

  @impl Loss
  def loss!(y, p) do
    p = Utils.clip(p, 1.0e-15, 1-1.0e-15)
    a = Matrex.apply(Matrex.multiply(y, Matrex.apply(p, :log)), fn x -> -1 * x end)
    b = Matrex.multiply(Matrex.subtract(1, y), Matrex.apply(Matrex.subtract(1, p), :log))
    Matrex.subtract(a, b)
  end
  
  @impl Loss
  def gradient!(y, p) do
    p = Utils.clip(p, 1.0e-15, 1-1.0e-15)
    a = Matrex.apply(Matrex.divide(y, p), fn x -> -1 * x end)
    b = Matrex.divide(Matrex.subtract(1, y), Matrex.subtract(1, p))
    Matrex.add(a, b)
  end

  def cross_entropy(%{}) do
    Agent.start_link(fn -> %CrossEntropy{} end)
  end
end        

