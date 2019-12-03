import Matrex
import Optimizer
import Utils
import Numex

alias Matrex
alias Dense
alias Utils

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
    Numex.add(dot(x, w), b)
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
    Numex.multiply(Activations.gradient(act_fn, layer_input), accum_grad)
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
  
  @behaviour Layer
  
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
  defstruct [:mask, :input_shape, :n, p: 0.2, pass_through: true, trainable: true]
  
  @behaviour Layer
 
  @impl Layer
  def forward_propogate(dropout_layer, m) do
    train? = get(dropout_layer, :trainable)
    prob = get(dropout_layer, :p)
    
    if train? do
      {rows, cols} = Matrex.size(m)
      masker = fn -> if :rand.uniform() > prob, do: 1, else: 0 end
      masked = Matrex.new(rows, cols, masker)
      put(dropout_layer, :mask, masked)
    end  
    
    c = if train?, do: get(dropout_layer, :mask), else: 1 - prob
    Numex.multiply(m, c)
  end
  
  @impl Layer
  def backward_propogate(dropout_layer, accum_grad) do
    Numex.multiply(accum_grad, get(dropout_layer, :mask))
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
  
  def dropout(%{}) do
    Agent.start_link(fn -> %Dropout{} end)
  end 
end

defmodule BatchNormalization do
  defstruct [:gamma, :beta, :running_mean, :gamma_opt, :beta_opt, :running_var, 
             :x_centered, :stddev_inv, momentum: 0.99, eps: 0.01, trainable: true]
  
  @behaviour Layer
  
  @impl Layer
  def init!(bn_layer, optimizer) do
    {rows, cols} = get(bn_layer, :shape_input)
    put(bn_layer, :gamma, Matrex.ones(rows, cols))
    put(bn_layer, :beta, Matrex.zeros(rows, cols))
    put(bn_layer, :gamma_opt, Optimizer.copy(optimizer))
    put(bn_layer, :beta_opt, Optimizer.copy(optimizer))
  end
  
  @impl Layer
  def parameters(bn_layer) do
    prod = fn ({rows, cols} = _shape) -> rows * cols end
    prod.(Matrex.size(get(bn_layer, :gamma))) + prod.(Matrex.size(get(bn_layer, :beta)))
  end
  
  @impl Layer
  def forward_propogate(bn_layer, m) do
    training = true # hack delete
    if get(bn_layer, :running_mean) == nil do
      put(bn_layer, :running_mean, Utils.mean_of_cols(m))
      put(bn_layer, :running_var, Utils.variance_of_cols(m))
    end
    
    {mean, var} = 
      if training and get(bn_layer, :trainable) do
        mean = Utils.mean_of_cols(m)
        var = Utils.variance_of_cols(m)
        put(bn_layer, :running_mean, Numex.add(Numex.multiply(get(bn_layer, :momentum), get(bn_layer, :running_mean)), 
                                               Numex.multiply((1 - get(bn_layer, :momentum)), mean)))
        put(bn_layer, :running_var, Numex.add(Numex.multiply(get(bn_layer, :momentum), get(bn_layer, :running_var)),
                                              Numex.multiply((1 - get(bn_layer, :momentum)), var)))
        {mean, var}
      else
        mean = get(bn_layer, :running_mean)
        var = get(bn_layer, :running_var)
        {mean, var}
      end
      
    put(bn_layer, :x_centered, Matrex.apply(m, fn x, _, c -> x - Matrex.at(mean, 1, c) end))
    put(bn_layer, :stddev_inv, Matrex.apply(Numex.add(var, get(bn_layer, :eps)), fn x -> 1 / :math.sqrt(x) end))
    stddev_inv = get(bn_layer, :stddev_inv) 
    x_norm = Numex.multiply(get(bn_layer, :x_centered), stddev_inv)
    Numex.add(Numex.multiply(get(bn_layer, :gamma), x_norm), get(bn_layer, :beta))
  end
  
  @impl Layer
  def backward_propogate(bn_layer, accum_grad) do
    {trainable, x_centered, stddev_inv, gamma, gamma_opt, beta, beta_opt} =
      Enum.reduce([:trainable, :x_centered, :stddev_inv, :gamma, :gamma_opt, :beta, :beta_opt], 
        {}, fn key, vals -> Tuple.append(vals, get(bn_layer, key)) end)

    if trainable do
      x_norm = Numex.multiply(x_centered, stddev_inv)
      grad_gamma = Utils.sum_of_cols(Numex.multiply(x_norm, accum_grad))
      grad_beta = Utils.sum_of_cols(accum_grad)
      put(bn_layer, :gamma, Optimizer.update(gamma_opt, gamma, grad_gamma))
      put(bn_layer, :beta, Optimizer.update(beta_opt, beta, grad_beta))
    end
    
    {batch_size, _} = Matrex.size(accum_grad)
      
    accum_grad = 
      (1 / batch_size)                                                        # (1 / batch_size)
      |> (fn(x) -> Numex.multiply(x, gamma) end).()                           # * gamma
      |> (fn(x) -> Numex.multiply(x, stddev_inv) end).()                      # * stddev_inv
      |> (fn(x) -> Numex.multiply(x, batch_size                               # * ( batch_size
        |> (fn(x) -> Numex.multiply(x, accum_grad) end).()                    #     * accum_grad
        |> (fn(x) -> Numex.subtract(x, Utils.sum_of_cols(accum_grad)) end).() #     * sum_of_cols(accum_grad)
        |> (fn(x) -> Numex.subtract(x, x_centered                             #     - ( x_centered
          |> (fn(x) -> Numex.multiply(x, :math.pow(stddev_inv)) end).()       #         * stddev_inv**2 
          |> (fn(x) -> Numex.multiply(x, Utils.sum_of_cols(                   #         * sum_of_cols(accum_grad * x_centered) ) )
            Numex.multiply(accum_grad, x_centered))) end).()         
        ) end).() 
      ) end).()           
  end
  
  @impl Layer
  def get(flatten_layer, key) do
    Agent.get(flatten_layer, &Map.get(&1, key))
  end
  
  @impl Layer
  def put(bn_layer, key, value) do
    Agent.update(bn_layer, &Map.put(&1, key, value))
  end
  
  def batchnorm(%{}) do
    Agent.start_link(fn -> %BatchNormalization{} end)
  end 
end

