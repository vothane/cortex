defmodule NeuralNetwork do
  
  defstruct [optimizer: nil, layers: [], loss_fn: nil, trainable?: true]
  
  def initialize(nn, optimizer, loss) do
    Agent.update(nn, fn state -> Map.put(state, :optimizer, optimizer) end)
    Agent.update(nn, fn state -> Map.put(state, :loss_fn, loss_fn) end)
  end
  
  def add(nn, layer) do
    layers = Agent.get(nn, fn state -> Map.get(state, :layers) end)
    layers_ = layers ++ layer
    Agent.update(nn, fn state -> Map.put(state, :layers, layers_) end)
  end
  
  defp forward_propogate(X) do
    layers = Agent.get(nn, fn state -> Map.get(state, :layers) end)
    Enum.reduce(layers, X, fn(layer, output) -> Layer.forward_propogate(layer, output) end)
  end  
end  