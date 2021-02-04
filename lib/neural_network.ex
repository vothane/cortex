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

  def set_trainable(nn, is_trainable?) do
    layers = Agent.get(nn, fn state -> Map.get(state, :layers) end)
    f = fn l -> %mod{} = Agent.get(l, &(&1)); mod end
    Enum.reduce(layers, nil, fn layer, _ -> f.(layer).put(layer, :trainable, is_trainable?) end)
  end

  def forward_propogate(nn, m) do
    layers = Agent.get(nn, fn state -> Map.get(state, :layers) end)
    f = fn l -> %mod{} = Agent.get(l, &(&1)); mod end
    Enum.reduce(layers, m, &(f.(&1).forward_propogate(&1, &2)))
  end  

  def backward_propogate(nn, loss_grad) do
    layers = Agent.get(nn, fn state -> Map.get(state, :layers) end)
    f = fn l -> %m{} = Agent.get(l, &(&1)); m end         
    Enum.reduce(Enum.reverse(layers), loss_grad, &(f.(&1).backward_propogate(&1, &2)))
  end 

  def fit(nn, x_data, y_data, epochs) do
    loss_fn = get(nn, :loss_fn)
    data = Enum.zip(x_data, y_data)
    
    Enum.reduce_while(1..epochs, nil, fn(epoch, _) ->
      if epoch <= epochs do
        Enum.reduce(data, nil, fn({x, y_true}, _) ->
          y_pred = NeuralNetwork.forward_propogate(nn, x)
          loss = Loss.loss(loss_fn, y_true, y_pred)
          NeuralNetwork.backward_propogate(nn, loss)
        end)
        {:cont, epoch + 1}
      else
        {:halt, true}
      end
    end)
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
  
  def neural_network(optimizer, loss) do
    Agent.start_link(fn -> %NeuralNetwork{optimizer: optimizer, loss_fn: loss} end)
  end      
end  

