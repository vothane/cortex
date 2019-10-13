defmodule NeuralNetwork do
  
  defstruct [optimizer: nil, layers: [], loss_fn: nil, trainable?: true]
  
  def initialize(nn, optimizer, loss_fn) do
    Agent.update(nn, fn state -> Map.put(state, :optimizer, optimizer) end)
    Agent.update(nn, fn state -> Map.put(state, :loss_fn, loss_fn) end)
  end
  
  def add(nn, layer) do
    layers = Agent.get(nn, fn state -> Map.get(state, :layers) end)
    optimizer = Agent.get(nn, fn state -> Map.get(state, :optimizer) end)
    %module{} = layer
    
    if List.last(layers) do 
      last = List.last(layers)
      module.set_input_shape(module.output_shape(last))
    end
    
    if function_exported?(module, :init, 3) do
      module.init(layer, optimizer)
    end
    
    layers_ = layers ++ layer
    Agent.update(nn, fn state -> Map.put(state, :layers, layers_) end)
  end
  
  defp forward_propogate(nn, X) do
    layers = Agent.get(nn, fn state -> Map.get(state, :layers) end)
    f = fn l -> %m{}; m end
    Enum.reduce(layers, X, &(f(&1).forward_propogate(&1, &2)))
  end  

  def backward_propogate(nn, loss_grad) do
    layers = Agent.get(nn, fn state -> Map.get(state, :layers) end)
    f = fn l -> %m{}; m end         
    Enum.reduce(Enum.reverse(layers), loss_grad, &(f(&1).backward_propogate(&1, &2)))
  end  
end  

