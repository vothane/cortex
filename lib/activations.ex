defmodule Activations do
  @callback gradient!(m)
  
  def gradient(activation, m) do
    %module{} = activation
    module.gradient!(activation,m)   
  end
end

defmodule Sigmoid do
  defstruct [sigmoid: fn x -> 1.0 / (1.0 + :math.exp(-x)) end]

  def gradient!(sigmoid, m) do # derivative of sigmoid
    f = Map.get(sigmoid, :sigmoid)
    dfx = &(f.(&1) * (1.0 - f.(&1)))
    Matrex.apply(m, dfx)
  end    
end

defmodule TanH do
  defstruct[tanh: &(:math.tanh(&1))]

  def gradient!(tanh, m) do # derivative of tanh
    f = Map.get(tanh, :tanh)
    dfx = &(1 - :math.pow(f.(&1), 2))
    Matrex.apply(m, dfx)
  end
end  

defmodule ReLU do
  defstruct[relu: fn x -> if x >= 0, do: x, else: 0 end]

  def gradient!(relu, m) do
    Matrex.apply(m, &(if &1 >= 0, do: 1, else: 0)])
  end
end  