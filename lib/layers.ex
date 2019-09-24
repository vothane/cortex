defmodule Layer do
  @callback shape(layer, rows :: Integer, cols :: Integer) :: {:ok, {rows, cols}} | {:error, String.t}
  @callback layer_name(layer, String.t) :: String.t
  @callback parameters(layer, any) :: any
  @callback forward_propogate(layer, any) :: any 
  @callback backward_propogate(layer, any) :: any 
  @callback output_shape(layer) :: any
end

defmodule Dense do

  import Matrex

  defstruct [shape: nil, weights: nil, limit: nil, name: nil]
  
  @behaviour Layer

  def shape(dense_layer, shape), do: Agent.update(dense_layer, fn state -> Map.put(state, :shape, shape) end)
  def layer_name(dense_layer, name), do: Agent.update(dense_layer, fn state -> Map.put(state, :name, name) end)
  def parameters(dense_layer, args), do: nil
  def forward_propogate(dense_layer), do: nil
  def backward_propogate(dense_layer), do: nil
  def output_shape(dense_layer) :: Matrex.size(Agent.get(dense_layer, fn state -> state.weights end))
  
  def initialize(dense_layer, optimizer, init_fun // &:rand.uniform/1) do
    limit = 1 / sqrt(input_shape[0])
    Agent.update(dense_layer, fn state -> Map.put(state, :limit, limit) end)
    Agent.update(dense_layer, fn state -> Map.put(state, :weights, Matrex.new(rows, columns, init_fun)) end)
  end 
  
  def new() do
    %Dense{shape: nil, weights: nil, limit: nil}
  end      
  
  def start_link(), do: Agent.start_link(fn -> Dense.new() end)
end

defmodule Activation do

  import Matrex

  defstruct [shape: nil, activation_fn: nil, name: nil]
  
  @behaviour Layer

  def shape(activation_layer, shape), do: nil
  def layer_name(activation_layer, name), do: Agent.update(activation_layer, fn state -> Map.put(state, :name, name) end)
  def parameters(activation_layer, args), do: nil
  def forward_propogate(activation_layer, X), do: Agent.get(activation_layer, fn state -> state.activation_fn end).(X)
  def backward_propogate(activation_layer), do: nil
  def output_shape(activation_layer) :: Agent.get(activation_layer, fn state -> state.shape end)
  
  def initialize(activation_layer, activation_fn) do
    Agent.update(dense_layer, fn state -> Map.put(state, :activation_fn, activation_fn) end)
  end 
  
  def new() do
    %Activation{[activation_fn: nil, name: nil}
  end      
  
  def start_link(), do: Agent.start_link(fn -> Activation.new() end)
end
