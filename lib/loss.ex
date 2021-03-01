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
  def loss!(y, y_pred), do: Nx.multiply(0.5, Nx.power(Nx.subtract(y, y_pred), 2))
  
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
    a = Nx.multiply(y, Nx.log(p))
    b = Nx.subtract(1, p)
    c = Nx.multiply(Nx.subtract(1, y), Nx.log(b))
    Nx.map(Nx.subtract(a, c), fn x -> -1 * x end)
  end
  
  @impl Loss
  def gradient!(y, p) do
    p = Nx.clip(p, 1.0e-15, 1-1.0e-15)
    a = Nx.divide(y, p)
    b = Nx.divide(Nx.subtract(1, y), Nx.subtract(1, p))
    Nx.map(Nx.add(a, b), fn x -> -1 * x end)
  end

  def cross_entropy(%{}) do
    Agent.start_link(fn -> %CrossEntropy{} end)
  end
end        

