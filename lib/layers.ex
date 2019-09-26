defmodule Layer do
  @callback shape(struct, integer, integer) :: {:ok, {integer, integer}} | {:error, String.t}
  @callback layer_name(struct, String.t) :: String.t
  @callback parameters(struct, any) :: any
  @callback forward_propogate(struct, any) :: any 
  @callback backward_propogate(struct, any) :: any 
  @callback output_shape(struct) :: any
end

defmodule Dense do

  import Matrex

  defstruct [shape: nil, weights: nil, name: nil, input: nil]
  
  @behaviour Layer

  def shape(dense_layer, shape), do: Agent.update(dense_layer, fn state -> Map.put(state, :shape, shape) end)
  
  def layer_name(dense_layer, name), do: Agent.update(dense_layer, fn state -> Map.put(state, :name, name) end)
  
  def parameters(dense_layer, args), do: nil
  
  def forward_propogate(dense_layer, X) do 
    Agent.update(dense_layer, fn state -> Map.put(state, :input, X) end)
    W = Agent.get(dense_layer, fn state -> Map.get(state, :weights) end)
    Matrex.dot(W, X)
  end
  
  def backward_propogate(dense_layer, accum_grad) do
    W = Agent.get(dense_layer, fn state -> Map.get(state, :weights) end)
    input = Agent.get(dense_layer, fn state -> Map.get(state, :input) end)
    grad_w = Matrex.dot_tn(input, accum_grad)
    grad_w0 = accum_grad
           |> Matrex.to_list_of_lists()
           |> Enum.map(&Matrex.sum/1)
           |> Matrex.new()
           |> Matrex.transpose()
    accum_grad = Matrex.dot_nt(accum_grad, W)
    accum_grad
  end
  
  def output_shape(dense_layer), do: Matrex.size(Agent.get(dense_layer, fn state -> state.weights end))
  
  def initialize(dense_layer, optimizer, init_fn \\ &:rand.uniform/1) do
    {rows, columns} = Agent.get(dense_layer, fn state -> Map.get(state, :shape) end)
    Agent.update(dense_layer, fn state -> Map.put(state, :weights, Matrex.new(rows, columns, init_fn)) end)
  end 
  
  def new() do
    %Dense{shape: nil, weights: nil}
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
  def output_shape(activation_layer), do: Agent.get(activation_layer, fn state -> state.shape end)
  
  def initialize(activation_layer, activation_fn) do
    Agent.update(activation_layer, fn state -> Map.put(state, :activation_fn, activation_fn) end)
  end 
  
  def new() do
    %Activation{activation_fn: nil, name: nil}
  end      
  
  def start_link(), do: Agent.start_link(fn -> Activation.new() end)
end
