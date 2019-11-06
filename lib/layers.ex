defmodule Layer do
  @type dim :: {integer, integer}
  @callback set_input_shape!(struct, dim) :: {:ok, {integer, integer}} | {:error, String.t}
  @callback get_output_shape!(struct) :: any
  @callback layer_name(struct, String.t) :: String.t
  @callback parameters(struct, any) :: any
  @callback forward_propogate(struct, any) :: any 
  @callback backward_propogate(struct, any) :: any 
  @callback init!(struct, any) :: any
  @callback get(struct, atom) :: any
  @callback put(struct, atom, any) :: any
  
  def set_input_shape(layer, shape) do
    %module{} = Agent.get(layer, &(&1))
    module.set_input_shape!(layer, shape)
  end
  
  def get_output_shape(layer) do
    %module{} = Agent.get(layer, &(&1))
    module.get_output_shape!(layer)
  end
  
  def init(layer, params) do
    %module{} = Agent.get(layer, &(&1))
    if function_exported?(module, :init!, 2) do
      module.init!(layer, params)
      true
    else
      false
    end  
  end
end

defmodule Dense do
  defstruct [:layer_input, :shape_input, :n, :weights, :bias, :w_opt, :bias_opt, :output_shape, trainable: true]
  @enforce_keys [:n]
  
  import Matrex
  import Optimizer
  import Utils
  
  alias Matrex
  alias Dense
  alias Utils
  
  @behaviour Layer
  
  @impl Layer
  def set_input_shape!(dense_layer, shape_input), do: put(dense_layer, :shape_input, shape_input)
  
  @impl Layer
  def get_output_shape!(dense_layer), do: {1, get(dense_layer, :n)}
 
  @impl Layer
  def layer_name(dense_layer, name), do: put(dense_layer, :name, name)
 
  @impl Layer
  def parameters(dense_layer, args), do: nil
  
  @impl Layer
  def forward_propogate(dense_layer, x) do 
    put(dense_layer, :layer_input, x)
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
  def init!(dense_layer, optimizer) do
    put(dense_layer, :weights, Matrex.new(elem(get(dense_layer, :shape_input), 1), get(dense_layer, :n), fn -> :rand.uniform() end))
    put(dense_layer, :bias, zeros(1, get(dense_layer, :n)))
    put(dense_layer, :w_opt, Optimizer.copy(optimizer))
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
  
  def dense(params) do
    sin = Map.get(params, :shape_input, nil)
    n = Map.get(params, :n)
    {status, dense_layer} = Agent.start_link(fn -> %Dense{shape_input: sin, n: n} end)  
    dense_layer
  end 
end

defmodule Activation do
  defstruct [:activation_fn, :input, :name, trainable: true]
  @enforce_keys [:activation_fn]
  
  @activation_functions %{sigmoid: %Sigmoid{}, tanh: %TanH{}, relu: %ReLU{}}

  import Matrex
  import Utils
  
  @behaviour Layer
  
  @impl Layer
  def set_input_shape!(activation_layer, shape_input), do: put(activation_layer, :shape_input, shape_input)
  
  @impl Layer
  def get_output_shape!(activation_layer) do
    get(activation_layer, :shape_input)
  end
  
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
  def get(activation_layer, key) do
    Agent.get(activation_layer, &Map.get(&1, key))
  end
  
  @impl Layer
  def put(activation_layer, key, value) do
    Agent.update(activation_layer, &Map.put(&1, key, value))
  end
  
  def activation(activation) do
    Agent.start_link(fn -> %Activation{activation_fn: Map.get(@activation_functions, activation)} end)
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
    {rows, cols} = get(flatten_layer, :prev_shape)
    Matrex.reshape(accum_grad, rows, cols)
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

defmodule Dropout do
  defstruct [:p, :mask, :input_shape, :n, pass_through: true, trainable: true]
 
  @impl Layer
  def forward_propogate(dropout_layer, m) do
    train? = get(dropout_layer, :trainable)
    prob = get(dropout_layer, :p)
    c = 1 - get(dropout_layer, :p)
    
    if train? do
      {rows, cols} = Matrex.size(m)
      masker = fn el -> if :rand.uniform() > prob, do: 1, else: 0 end
      masked = Matrex.new(rows, cols, masker)
      put(dropout_layer, :mask, masked)
      c = masked
    end  
    
    Matrex.multiply(m, c)
  end
  
  @impl Layer
  def backward_propogate(dropout_layer, accum_grad) do
    Matrex.multiply(accum_grad, get(dropout_layer, :mask))
  end
  
  @impl Layer
  def output_shape(dropout_layer) do
    get(dropout_layer, :input_shape)
  end
  
  @impl Layer
  def get(dropout_layer, key) do
    Agent.get(dropout_layer, &Map.get(&1, key))
  end
  
  @impl Layer
  def put(dropout_layer, key, value) do
    Agent.update(dropout_layer, &Map.put(&1, key, value))
  end
  
  def dropout(%{input_shape: input_shape}) do
    Agent.start_link(fn -> %Dropout{input_shape: input_shape} end)
  end 
end

