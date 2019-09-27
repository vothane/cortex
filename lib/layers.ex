defmodule Layer do
  @callback shape_input(struct, integer, integer) :: {:ok, {integer, integer}} | {:error, String.t}
  @callback layer_name(struct, String.t) :: String.t
  @callback parameters(struct, any) :: any
  @callback forward_propogate(struct, any) :: any 
  @callback backward_propogate(struct, any) :: any 
  @callback output_shape(struct) :: any
end

defmodule Dense do

  import Matrex

  defstruct [input: nil, shape_input: nil, n: nil, weights: nil, bias: nil, name: nil]
  
  @behaviour Layer

  def shape_input(dense_layer, shape_input), do: Agent.update(dense_layer, fn state -> Map.put(state, :shape_input, shape_input) end)
  
  def layer_name(dense_layer, name), do: Agent.update(dense_layer, fn state -> Map.put(state, :name, name) end)
  
  def parameters(dense_layer, args), do: nil
  
  def forward_propogate(dense_layer, X) do 
    Agent.update(dense_layer, fn state -> Map.put(state, :input, X) end)
    W = Agent.get(dense_layer, fn state -> Map.get(state, :weights) end)
    Matrex.add(Matrex.dot(W, X), Agent.get(dense_layer, fn state -> Map.get(state, :bias) end)) 
  end
  
  def backward_propogate(dense_layer, accum_grad) do
    W = Agent.get(dense_layer, fn state -> Map.get(state, :weights) end)
    input = Agent.get(dense_layer, fn state -> Map.get(state, :input) end)
    grad_w = Matrex.dot_tn(input, accum_grad)
    Agent.update(dense_layer, fn state -> Map.put(state, :weights, grad_w) end)
    grad_bias = accum_grad
           |> Matrex.to_list_of_lists()
           |> Enum.map(&Matrex.sum/1)
           |> Matrex.new()
           |> Matrex.transpose()
    Agent.update(dense_layer, fn state -> Map.put(state, :bias, grad_bias) end)
    accum_grad = Matrex.dot_nt(accum_grad, W) # loss
    accum_grad
  end
  
  def output_shape_input(dense_layer), do: Matrex.size(Agent.get(dense_layer, fn state -> state.weights end))
  
  def initialize(dense_layer, optimizer, init_fn \\ &:rand.uniform/1) do
    {rows, columns} = Agent.get(dense_layer, fn state -> Map.get(state, :shape_input) end)
    Agent.update(dense_layer, fn state -> Map.put(state, :weights, Matrex.new(rows, columns, init_fn)) end)
  end 
  
  def dense(opts) do
    f = fn coll -> &(Keyword.get(coll, &1)) end # fuck elixir ur a fp lang act like 1 -> partial fns
    g = f.(opts) # curry into f

    %Dense{shape_input: g.(:shape_input), 
           n: g.(:n),
           weights: Matrex.new(elem(g.(:shape_input), 0), g.(:n)),
           bias: Matrex.zeros(1, g.(:n))
          }
  end      
  
  def start_link(), do: Agent.start_link(fn -> Dense.new() end)

  
end

defmodule Activation do

  import Matrex

  defstruct [shape_input: nil, activation_fn: nil, name: nil]
  
  @behaviour Layer

  def shape_input(activation_layer, shape_input), do: nil
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
