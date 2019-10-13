defmodule Activations do
  @type matrex :: %Matrex{data: binary}
  
  @callback activate!(struct, matrex) :: matrex
  @callback gradient!(struct, matrex) :: matrex
  
  def activate(activation, m) do
    %module{} = activation
    module.acivate!(activation,m)   
  end
  
  def gradient(activation, m) do
    %module{} = activation
    module.gradient!(activation,m)   
  end
end

defmodule Sigmoid do
  defstruct [sigmoid: &Sigmoid.sigmoid/1]
  
  @behaviour Activations
  
  @impl Activations
  def activate!(sigmoid, m) do
    f = Map.get(sigmoid, :sigmoid)
    Matrex.apply(m, f)
  end
  
  @impl Activations
  def gradient!(sigmoid, m) do # derivative of sigmoid
    f = Map.get(sigmoid, :sigmoid)
    dfx = &(f.(&1) * (1.0 - f.(&1)))
    Matrex.apply(m, dfx)
  end    
  
  defp sigmoid(x), do: 1.0 / (1.0 + :math.exp(-x)) 
end

defmodule TanH do
  defstruct [tanh: &(:math.tanh(&1))]
  
  @behaviour Activations
  
  @impl Activations
  def activate!(tanh, m) do
    f = Map.get(tanh, :tanh)
    Matrex.apply(m, f)
  end
  
  @impl Activations
  def gradient!(tanh, m) do # derivative of tanh
    f = Map.get(tanh, :tanh)
    dfx = &(1 - :math.pow(f.(&1), 2))
    Matrex.apply(m, dfx)
  end
end  

defmodule ReLU do
  defstruct [relu: &ReLu.relu/1]
  
  @behaviour Activations
  
  @impl Activations
  def activate!(relu, m) do
    f = Map.get(relu, :relu)
    Matrex.apply(m, f)
  end

  @impl Activations
  def gradient!(relu, m) do
    Matrex.apply(m, &(if &1 >= 0, do: 1, else: 0))
  end
  
  defp relu(x), do: if x >= 0, do: x, else: 0
end  