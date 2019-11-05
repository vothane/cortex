defmodule Activations do
  @callback activate!(struct, any) :: any
  @callback gradient!(struct, any) :: any
  
  def activate(%module{} = activation, m) do
    module.activate!(activation, m)   
  end
  
  def gradient(%module{} = activation, m) do
    module.gradient!(activation, m)   
  end
end

defmodule Sigmoid do
  defstruct [name: :sigmoid]
  
  @behaviour Activations
  
  @impl Activations
  def activate!(sigmoid, m) do
    Matrex.apply(m, &sigmoid/1)
  end
  
  @impl Activations
  def gradient!(sigmoid, m) do # derivative of sigmoid
    dfx = &(sigmoid(&1) * (1.0 - sigmoid(&1)))
    Matrex.apply(m, dfx)
  end    
  
  def sigmoid(x), do: 1.0 / (1.0 + :math.exp(-x)) 
end

defmodule TanH do
  defstruct [name: :tanh]
  
  @behaviour Activations
  
  @impl Activations
  def activate!(tanh, m) do
    Matrex.apply(m, &tanh/1)
  end
  
  @impl Activations
  def gradient!(tanh, m) do # derivative of tanh
    dfx = &(1 - :math.pow(tanh(&1), 2))
    Matrex.apply(m, dfx)
  end
  
  def tanh(x), do: :math.tanh(x)
end  

defmodule ReLU do
  defstruct [name: :relu]
  
  @behaviour Activations
  
  @impl Activations
  def activate!(relu, m) do
    Matrex.apply(m, &relu/1)
  end

  @impl Activations
  def gradient!(relu, m) do
    Matrex.apply(m, &(if &1 >= 0, do: 1, else: 0))
  end
  
  def relu(x), do: if x >= 0, do: x, else: 0
end  

defmodule Softmax do
  defstruct [name: :softmax]
  
  @behaviour Activations
  
  @impl Activations
  def activate!(softmax, m) do
    maxima = Utils.max_of_rows(m)
    f = fn ({row, max}) -> Matrex.subtract(row, Matrex.scalar(max)) end
    diffs = Enum.map(Enum.zip(Matrex.list_of_rows(m), Matrex.list_of_rows(maxima)), f) 
    e_x = Matrex.apply(Matrex.new([diffs]), :exp) 
    sums = Utils.sum_of_rows(e_x)
    g = fn ({row, sum}) -> Matrex.divide(row, Matrex.scalar(sum)) end
    result = Enum.map(Enum.zip(Matrex.list_of_rows(e_x), Matrex.list_of_rows(sums)), g) 
    Matrex.new([result])
  end

  @impl Activations
  def gradient!(softmax, m) do
    p = Softmax.activate!(softmax, m)
    Matrex.multiply(p, Matrex.subtract(1, p))
  end
end  