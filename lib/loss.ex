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
  def loss!(y, y_pred), do: Nx.multiply(0.5, Nx.map(Nx.subtract(y, y_pred), fn x -> :math.pow(x, 2) end))
  
  @impl Loss
  def gradient!(y, y_pred), do: Nx.multiply(-1, (Nx.subtract(y, y_pred)))

  def square_loss(%{}) do
    Agent.start_link(fn -> %SquareLoss{} end)
  end
end

defmodule CrossEntropy do
  defstruct [name: :cross_entropy]
  
  @behaviour Loss

  @impl Loss
  def loss!(y, p) do
    # (- y * log(p)) - ((1 - y) * log(1 - p))
    # (a) - ((1 - y) * log(b))
    # (a) - (c)
    p = Nx.clip(p, 1.0e-15, 1-1.0e-15)
    a = Nx.map(Nx.multiply(y, Nx.log(p)), fn x -> -1 * x end)
    b = Nx.clip(Nx.subtract(1, p), 1.0e-15, 1 - 1.0e-15)
    c = Nx.multiply(Nx.subtract(1, y), Nx.log(b))
    Nx.subtract(a, c)
  end
  
  @impl Loss
  def gradient!(y, p) do
    p = Nx.clip(p, 1.0e-15, 1-1.0e-15)
    a = Nx.map(Nx.divide(y, p), fn x -> -1 * x end)
    b = Nx.divide(Nx.subtract(1, y), Nx.subtract(1, p))
    Nx.add(a, b)
  end

  def cross_entropy(%{}) do
    Agent.start_link(fn -> %CrossEntropy{} end)
  end
end        

