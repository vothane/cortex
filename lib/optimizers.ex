import Nx
import Utils

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
      put(sgd, :w_, Utils.zeros(Nx.shape(w)))
    end 
    w_ = get(sgd, :w_)
    learning_rate = get(sgd, :learning_rate)
    momentum = get(sgd, :momentum)
    w_ = Nx.add(Nx.multiply(w_, momentum), Nx.multiply(grad_wrt_w, (1 - momentum)))
    Nx.subtract(w, Nx.multiply(w_, learning_rate))
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
      put(rmsp, :run_avg, Utils.zeros(Nx.shape(grad_wrt_w)))
    end

    {learning_rate, run_avg, rho, eps} =
      {get(rmsp, :learning_rate), get(rmsp, :run_avg), get(rmsp, :rho), get(rmsp, :eps)}

    running_average =
      Nx.add(Nx.multiply(run_avg, rho), Nx.multiply(1 - rho, Nx.power(grad_wrt_w, 2)))

    put(rmsp, :run_avg, running_average)

    Nx.subtract(w,
      Nx.multiply(learning_rate,
        Nx.divide(
          grad_wrt_w,
          Nx.map(Nx.add(running_average, eps), &:math.sqrt/1))))
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

defmodule Adam do
  alias Adam

  defstruct [learning_rate: 0.001, eps: 1.0e-8, b1: 0.9, b2: 0.999, m: nil, v: nil]

  @behaviour Optimizer

  @impl Optimizer
  def update!(adam, w, grad_wrt_w) do
    if get(adam, :m) == nil do
      put(adam, :m, Utils.zeros(Nx.shape(grad_wrt_w)))
    end

    if get(adam, :v) == nil do
      put(adam, :v, Utils.zeros(Nx.shape(grad_wrt_w)))
    end

    {learning_rate, eps, b1, b2, m, v} =
      {get(adam, :learning_rate), get(adam, :eps), get(adam, :b1), get(adam, :b2), get(adam, :m), get(adam, :v)}

    m = Nx.add(Nx.multiply(b1, m), Nx.multiply((1 - b1), grad_wrt_w))
    v = Nx.add(Nx.multiply(b2, v), Nx.multiply((1 - b2), Nx.power(grad_wrt_w, 2)))
    put(adam, :m, m)
    put(adam, :v, v)

    m_hat = Nx.divide(m, (1 - b1))
    v_hat = Nx.divide(v, (1 - b2))

    w_ = Nx.divide(Nx.multiply(learning_rate, m_hat), Nx.add(Nx.map(v_hat, &:math.sqrt/1), eps))
    Nx.subtract(w, w_)
  end

  def adam(%{}) do
    Agent.start_link(fn -> %Adam{} end)
  end

  @impl Optimizer
  def get(adam, key) do
    Agent.get(adam, &Map.get(&1, key))
  end

  @impl Optimizer
  def put(adam, key, value) do
    Agent.update(adam, &Map.put(&1, key, value))
  end

  @impl Optimizer
  def copy!(adam) do
    {status, opt} =
    Agent.start_link(
      fn ->
        %Adam{learning_rate: get(adam, :learning_rate),
              eps: get(adam, :eps),
              b1: get(adam, :b1),
              b2: get(adam, :b2),
              m: get(adam, :m),
              v: get(adam, :v)}
      end)
    opt
  end
end    
