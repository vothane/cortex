defmodule Layer do
  @callback set_shape_input(struct, integer, integer) :: {:ok, {integer, integer}} | {:error, String.t}
  @callback layer_name(struct, String.t) :: String.t
  @callback parameters(struct, any) :: any
  @callback forward_propogate(struct, any) :: any 
  @callback backward_propogate(struct, any) :: any 
  @callback output_shape(struct) :: any
  @callback init(struct, struct) :: any
  @callback get(struct, atom) :: any
  @callback put(struct, atom, any) :: any
end

defmodule Dense do
  import Matrex
  import Optimizer
  
  alias Matrex
  alias Dense
  
  defstruct [layer_input: nil, shape_input: nil, n: nil, trainable: true, weights: nil, bias: nil, w_opt: nil, bias_opt: nil, output_shape: nil]
  
  @behaviour Layer
  
  @impl Layer
  def set_shape_input(dense_layer, shape_input), do: put(dense_layer, :shape_input, shape_input)
 
  @impl Layer
  def layer_name(dense_layer, name), do: put(dense_layer, :name, name)
 
  @impl Layer
  def parameters(dense_layer, args), do: nil
  
  @impl Layer
  def forward_propogate(dense_layer, X) do 
    put(dense_layer, :input, X)
    W = get(dense_layer, :weights)
    add(dot(W, X), get(dense_layer, :bias)) 
  end
  
  @impl Layer
  def backward_propogate(dense_layer, accum_grad) do 
    w = get(dense_layer, :weights)
    input = get(dense_layer, :layer_input)
    grad_w = dot_tn(input, accum_grad)
    put(dense_layer, :weights, grad_w)
    grad_bias = accum_grad
             |> transpose()
             |> to_list_of_lists()
             |> Enum.reduce([], fn x, acc -> acc ++ [Enum.sum(x)] end)
    put(dense_layer, :bias, Matrex.new([grad_bias]))
    Matrex.dot_nt(accum_grad, w)
  end
  
  @impl Layer
  def output_shape_input(dense_layer), do: size(get(dense_layer, :weights))
  
  @impl Layer
  def init(dense_layer, optimizer, init_fn \\ &(:rand.uniform(&1))) do
    put(dense_layer, :weights, Matrex.new(elem(get(dense_layer, :shape_input), 0), get(dense_layer, :n), fn -> :rand.uniform() end))
    put(dense_layer, :bias, zeros(get(dense_layer, :n)))
    put(dense_layer, :w_opt, optimizer)
    put(dense_layer, :bias_opt, Optimizer.copy(optimizer))
  end
  
  @impl Layer
  def get(dense_layer, key) do
    Agent.get(dense_layer, &Map.get(&1, key))
  end
  
  @impl Layer
  def put(dense_layer, key, value) do
    Agent.update(dense_layer, &Map.put(&1, key, value))
  end  
  
  def dense(%{shape_input: sin, n: n}) do
    Agent.start_link(fn -> %Dense{shape_input: sin, n: n} end)      
  end 
end

defmodule Activation do

  import Matrex

  defstruct [shape_input: nil, activation_fn: nil, name: nil]
  
  @behaviour Layer

  def set_shape_input(activation_layer, shape_input), do: nil
  def layer_name(activation_layer, name), do: Agent.update(activation_layer, fn state -> Map.put(state, :name, name) end)
  def parameters(activation_layer, args), do: nil
  def forward_propogate(activation_layer, X), do: Agent.get(activation_layer, fn state -> state.activation_fn end).(X)
  def backward_propogate(activation_layer), do: nil
  def output_shapet(activation_layer), do: Agent.get(activation_layer, fn state -> state.shape_input end)
  
  def initialize(activation_layer, activation_fn) do
    Agent.update(activation_layer, fn state -> Map.put(state, :activation_fn, activation_fn) end)
  end 
  
  def new() do
    %Activation{activation_fn: nil, name: nil}
  end      
  
  def start_link(), do: Agent.start_link(fn -> Activation.new() end)
end
