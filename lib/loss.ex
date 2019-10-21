defmodule Loss do
  @callback loss!(struct, any, any) :: any
  @callback gradient!(struct, any, any) :: any
  
  def loss(loss, actuals, predictions) do
    %module{} = loss
    module.loss!(loss, actuals, predictions)   
  end
  
  def gradient(loss, actuals, predictions) do
    %module{} = loss
    module.gradient!(loss, actuals, predictions)   
  end
end

defmodule SquareLoss do
  defstruct [name: :square_loss]
  
  @behaviour Loss
  
  @impl Loss
  def loss!(loss, y, y_pred), do: Matrex.multiply(0.5, Matrex.apply(Matrex.subtract(y, y_pred), fn x -> :math.pow(x, 2) end))
  
  @impl Loss
  def gradient!(loss, y, y_pred), do: Matrex.multiply(-1, (Matrex.subtract(y, y_pred)))
end

defmodule CrossEntropy do
  defstruct [name: :cross_entropy]
  
  @behaviour Loss

  @impl Loss
  def loss!(loss, y, p) do
    p = Utils.clip(p, 1.0e-15, 1-1.0e-15)
    a = Matrex.apply(Matrex.multiply(y, Matrex.apply(p, :log)), fn x -> -1 * x end)
    b = Matrex.multiply(Matrex.subtract(1, y), Matrex.apply(Matrex.subtract(1, p), :log))
    Matrex.subtract(a, b)
  end
  
  @impl Loss
  def gradient!(loss, y, p) do # - (y / p) + (1 - y) / (1 - p)
    p = Utils.clip(p, 1.0e-15, 1-1.0e-15)
    a = Matrex.apply(Matrex.divide(y, p), fn x -> -1 * x end)
    b = Matrex.divide(Matrex.subtract(1, y), Matrex.subtract(1, p))
    Matrex.add(a, b)
  end
end        
