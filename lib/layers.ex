defmodule Layer do
  @type dim :: {integer, integer}
  @callback set_shape_input(struct, dim) :: {:ok, {integer, integer}} | {:error, String.t}
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
  defstruct [layer_input: nil, shape_input: nil, n: nil, trainable: true, weights: nil, bias: nil, w_opt: nil, bias_opt: nil, output_shape: nil]
  
  import Matrex
  import Optimizer
  import Utils
  
  alias Matrex
  alias Dense
  alias Utils
  
  @behaviour Layer
  
  @impl Layer
  def set_shape_input(dense_layer, shape_input), do: put(dense_layer, :shape_input, shape_input)
 
  @impl Layer
  def layer_name(dense_layer, name), do: put(dense_layer, :name, name)
 
  @impl Layer
  def parameters(dense_layer, args), do: nil
  
  @impl Layer
  def forward_propogate(dense_layer, x) do 
    put(dense_layer, :input, x)
    w = get(dense_layer, :weights)
    b = get(dense_layer, :bias) 
    add_m_v(dot(x, w), b)
  end
  
  @impl Layer
  def backward_propogate(dense_layer, accum_grad) do 
    w = get(dense_layer, :weights)
    bias = get(dense_layer, :bias)
    input = get(dense_layer, :layer_input)
    grad_w = dot_tn(input, accum_grad)
    grad_bias = sum_of_cols(accum_grad)
    _w =  Optimizer.update(get(dense_layer, :w_opt), w, grad_w)
    put(dense_layer, :weights, _w)
    
    _bias = Optimizer.update(get(dense_layer, :bias_opt), bias, grad_bias)
    put(dense_layer, :bias, _bias)    
    
    Matrex.dot_nt(accum_grad, w)
  end
  
  @impl Layer
  def output_shape(dense_layer), do: size(get(dense_layer, :weights))
  
  @impl Layer
  def init(dense_layer, optimizer, init_fn \\ &(:rand.uniform(&1))) do
    put(dense_layer, :weights, Matrex.new(elem(get(dense_layer, :shape_input), 0), get(dense_layer, :n), fn -> :rand.uniform() end))
    put(dense_layer, :bias, zeros(1, get(dense_layer, :n)))
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
  defstruct [activation_fn: nil, input: nil, trainable: true, name: nil]
  
  import Matrex
  import Utils
  
  @behaviour Layer
  
  @impl Layer
  def forward_propogate(activation_layer, m) do
    put(activation_layer, :input, m)
    act_fn = get(activation_layer, :activation_fn)
    Activations.activate(act_fn, m)
  end
  
  @impl Layer
  def backward_propogate(activation_layer, accum_grad) do
    layer_input = get(activation_layer, :input)
    act_fn = get(activation_layer, :activation_fn)
    mult_m_v(Activations.gradient(act_fn, layer_input), accum_grad)
  end
  
  @impl Layer
  def output_shape(activation_layer) do
    Matrex.size(get(activation_layer, :input))
  end
  
  @impl Layer
  def get(activation_layer, key) do
    Agent.get(activation_layer, &Map.get(&1, key))
  end
  
  @impl Layer
  def put(activation_layer, key, value) do
    Agent.update(activation_layer, &Map.put(&1, key, value))
  end
  
  def activation(%{activation_fn: activation}) do
    Agent.start_link(fn -> %Activation{activation_fn: activation} end)
  end
end

defmodule Flatten do
  defstruct [prev_shape: nil, input_shape: nil, trainable: true, name: nil]
  
  @impl Layer
  def forward_propogate(flatten_layer, m) do
    {rows, cols} = shape_m = Matrex.size(m)
    put(flatten_layer, :prev_shape, shape_m)
    Matrex.reshape(m, 1, rows * cols)
  end
  
  @impl Layer
  def backward_propogate(flatten_layer, accum_grad) do
    Matrex.reshape(accum_grad, get(flatten_layer, :prev_shape))
  end
  
  @impl Layer
  def output_shape(flatten_layer) do
    {rows, cols} = Matrex.size(get(flatten_layer, :input_shape))
    {rows * cols, 1}
  end
  
  @impl Layer
  def get(flatten_layer, key) do
    Agent.get(flatten_layer, &Map.get(&1, key))
  end
  
  @impl Layer
  def put(flatten_layer, key, value) do
    Agent.update(flatten_layer, &Map.put(&1, key, value))
  end
  
  def flatten(%{input_shape: input_shape}) do
    Agent.start_link(fn -> %Flatten{input_shape: input_shape} end)
  end    
end

