defmodule NeuralNetwork do
  defstruct [optimizer: nil, layers: [], loss_fn: nil, trainable?: true]
  
  def initialize(nn, optimizer, loss_fn) do
    Agent.update(nn, fn state -> Map.put(state, :optimizer, optimizer) end)
    Agent.update(nn, fn state -> Map.put(state, :loss_fn, loss_fn) end)
  end
  
  def add(nn, layer) do
    layers = get(nn, :layers)
    optimizer = get(nn, :optimizer)
    
    if List.last(layers) do 
      last = List.last(layers)
      Layer.set_input_shape(layer, Layer.get_output_shape(last))
    end
    
    Layer.init(layer, optimizer)    
    layers = List.insert_at(layers, -1, layer)
    Agent.update(nn, fn state -> Map.put(state, :layers, layers) end)
  end
  
  def forward_propogate(nn, m) do
    layers = Agent.get(nn, fn state -> Map.get(state, :layers) end)
    f = fn l -> %mod{} = l; mod end
    Enum.reduce(layers, m, &(f.(&1).forward_propogate(&1, &2)))
  end  

  def backward_propogate(nn, loss_grad) do
    layers = Agent.get(nn, fn state -> Map.get(state, :layers) end)
    f = fn l -> %m{} = l; m end         
    Enum.reduce(Enum.reverse(layers), loss_grad, &(f.(&1).backward_propogate(&1, &2)))
  end    
  
  def get(nn, key) do
    Agent.get(nn, &Map.get(&1, key))
  end
  
  def put(nn, key, value) do
    Agent.update(nn, &Map.put(&1, key, value))
  end
  
  def neural_network(optimizer) do
    Agent.start_link(fn -> %NeuralNetwork{optimizer: optimizer} end)
  end    
end  

