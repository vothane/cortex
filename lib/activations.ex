defmodule Activations do
  @callback activate!(struct, any) :: any
  @callback gradient!(struct, any) :: any
  
  def activate(activation, m) do
    %module{} = activation
    module.activate!(activation, m)   
  end
  
  def gradient(activation, m) do
    %module{} = activation
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
  defstruct [relu: :relu]
  
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