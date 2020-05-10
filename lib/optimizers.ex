import Numex

defmodule Optimizer do
  @callback update!(struct, any, any) :: any
  @callback get(struct, atom) :: any
  @callback put(struct, atom, any) :: any
  @callback copy!(struct) :: any
  
  def copy(optimizer) do
    %module{}  = Agent.get(optimizer, &(&1))
    module.copy!(optimizer) 
  end
  
  def update(optimizer, w, grad_wrt_w) do
    %module{} = Agent.get(optimizer, &(&1))
    module.update!(optimizer, w, grad_wrt_w) 
  end
end
  
defmodule SGD do # Stochastic Gradient Descent
  alias SGD
  
  defstruct [learning_rate: nil, momentum: nil, w_: nil]
 
  @behaviour Optimizer
  
  @impl Optimizer
  def update!(sgd, w, grad_wrt_w) do # wrt with respect to (partial derivatives)
    if get(sgd, :w_) == nil do
      {rows, cols} = Matrex.size(w)
      put(sgd, :w_, Matrex.zeros(rows, cols))
    end 
    w_ = get(sgd, :w_)
    learning_rate = get(sgd, :learning_rate)
    momentum = get(sgd, :momentum)
    w_ = Numex.add(Matrex.multiply(w_, momentum), Matrex.multiply(grad_wrt_w, (1 - momentum)))
    Matrex.subtract(w, Matrex.multiply(w_, learning_rate))
  end
  
  def sgd(%{w_: w_, momentum: m, learning_rate: lr}) do
    Agent.start_link(fn -> %SGD{w_: w_, momentum: m, learning_rate: lr} end)      
  end
  
  @impl Optimizer
  def get(sgd, key) do
    Agent.get(sgd, &Map.get(&1, key))
  end
  
  @impl Optimizer
  def put(sgd, key, value) do
    Agent.update(sgd, &Map.put(&1, key, value))
  end
  
  @impl Optimizer
  def copy!(sgd) do
    {status, opt} =
    Agent.start_link(fn -> 
      %SGD{w_: get(sgd, :w_), 
           momentum: get(sgd, :momentum), 
           learning_rate: get(sgd, :learning_rate)} 
    end)
    opt
  end
end

defmodule RMSprop do # Root Mean Square Propagation
  alias RMSprop

  defstruct [learning_rate: 0.01, run_avg: nil, eps: 1.0e-8, rho: 0.9]

  @behaviour Optimizer

  @impl Optimizer
  def update!(rmsp, w, grad_wrt_w) do # wrt with respect to (partial derivatives)
    if get(rmsp, :run_avg) == nil do
      {rows, cols} = Matrex.size(w)
      put(rmsp, :run_avg, Matrex.zeros(rows, cols))
    end

    {learning_rate, run_avg, rho, eps} =
      {get(rmsp, :learning_rate), get(rmsp, :run_avg), get(rmsp, :rho), get(rmsp, :eps)}

    running_average =
      Numex.add(Matrex.multiply(run_avg, rho),
                Matrex.multiply(1 - rho, Matrex.apply(grad_wrt_w, fn x -> :math.pow(x, 2) end)))

    put(rmsp, :run_avg, running_average)

    #w - (learning_rate * grad_wrt_w / sqrt(running_average + eps)))
    Matrex.subtract(w,
      Matrex.multiply(learning_rate,
        Matrex.divide(
          grad_wrt_w,
          Matrex.apply(Matrex.add(running_average, eps), &:math.sqrt/1))))
  end

  def rmsp(%{}) do
    Agent.start_link(fn -> %RMSprop{} end)
  end

  @impl Optimizer
  def get(rmsp, key) do
    Agent.get(rmsp, &Map.get(&1, key))
  end

  @impl Optimizer
  def put(rmsp, key, value) do
    Agent.update(rmsp, &Map.put(&1, key, value))
  end

  @impl Optimizer
  def copy!(rmsp) do
    {status, opt} =
    Agent.start_link(
      fn ->
        %RMSprop{learning_rate: get(rmsp, :learning_rate),
                 run_avg: get(rmsp, :run_avg),
                 eps: get(rmsp, :eps),
                 rho: get(rmsp, :rho)}
      end)
    opt
  end
end    
