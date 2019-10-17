defmodule Loss do
  @callback loss!(struct, any, any) :: any
  @callback gradient!(struct, any, any) :: any
  
  def loss(loss, y, actual) do
    %module{} = loss
    module.loss!(loss, y, actual)   
  end
  
  def gradient(loss, y, actual) do
    %module{} = loss
    module.gradient!(loss, y, actual)   
  end
end

defmodule SquareLoss do
  defstruct [name: :square_loss]

  def loss!(loss, y, y_pred), do: Matrex.multiply(0.5, Matrex.apply(Matrex.subtract(y, y_pred), fn x -> :math.pow(x, 2)))

  def gradient!(loss, y, y_pred), do: Matrex.multiply(-1, (Matrex.subtract(y, y_pred)))
end

defmodule CrossEntropy do
  defstruct [name: :cross_entropy]

  def loss!(loss, y, p) do
    a = Matrex.multiply(y, Matrex.apply(p, :log))
    b = Matrex.multiply(Matrex.subtract(1, y), Matrex.apply(Matrex.subtract(1, p), :log))
    Matrex.multiply(-1, Matrex.subtract(a, b))
  end
  
  def gradient!(loss, y, p) do
    a = Matrex.divide(y, p)
    g = Matrex.subtract(1, y)
    h = Matrex.subtract(1, p)
    b = Matrex.divide(g, h)
    Matrex.multiply(-1, Matrex.add(a, b))
  end
end        

