defmodule Optimizer do
  @callback update!(struct, any, any) :: any
  @callback get(struct, atom) :: any
  @callback put(struct, atom, any) :: any
  @callback copy!(struct) :: any
  
  def copy(optimizer) do
    opt = Agent.get(optimizer, &(&1))
    %module{} = opt
    module.copy!(optimizer) 
  end
  
  def update(optimizer, w, grad_wrt_w) do
    opt = Agent.get(optimizer, &(&1))
    %module{} = opt
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
    w_ = Matrex.add(Matrex.multiply(w_, momentum), Matrex.multiply(grad_wrt_w, (1 - momentum)))
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
