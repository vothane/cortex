defmodule Activations do
  @callback activate(struct, any) :: any
  @callback gradient(struct, any) :: any
end

defprotocol Activations do
  def activate(activation, t)
  def gradient(activation, t)
end
  
defmodule Sigmoid do
  defstruct [name: :sigmoid]
  
  @behaviour Activations
  
  defimpl Activations do
    @impl Activations
    def activate(sigmoid, t) do
      Nx.map(t, &Sigmoid.sigmoid/1)
    end
  
    @impl Activations
    def gradient(sigmoid, t) do # derivative of sigmoid
      dfx = &(Sigmoid.sigmoid(&1) * (1.0 - Sigmoid.sigmoid(&1)))
      Nx.map(t, dfx)
    end    
  end
  
  def sigmoid(x), do: 1.0 / (1.0 + :math.exp(-x)) 
end

defmodule TanH do
  defstruct [name: :tanh]
  
  @behaviour Activations
  
  defimpl Activations do
    @impl Activations
    def activate(tanh, t) do
      Nx.map(t, &TanH.tanh/1)
    end
  
    @impl Activations
    def gradient(tanh, t) do # derivative of tanh
      dfx = &(1 - :math.pow(TanH.tanh(&1), 2))
      Nx.map(t, dfx)
    end
  end
  
  def tanh(x), do: :math.tanh(x)
end  

defmodule ReLU do
  defstruct [name: :relu]
  
  @behaviour Activations
  
  defimpl Activations do
    @impl Activations
    def activate(relu, t) do
      Nx.map(t, &ReLU.relu/1)
    end

    @impl Activations
    def gradient(relu, t) do
      Nx.map(t, &(if &1 >= 0, do: 1, else: 0))
    end
  end
  
  def relu(x), do: if x >= 0, do: x, else: 0
end  

defmodule LeakyReLU do
  defstruct [name: :leaky_relu, alpha: 0.2]
  
  @behaviour Activations
  
  defimpl Activations do
    @impl Activations
    def activate(lrelu, t) do
      Nx.map(t, fn x -> LeakyReLU.leaky_relu_fp(x, lrelu.alpha) end)
    end

    @impl Activations
    def gradient(lrelu, t) do
      Nx.map(t, fn x -> LeakyReLU.leaky_relu_bp(x, lrelu.alpha) end)
    end
  end
  
  def leaky_relu_fp(x, alpha), do: if x >= 0, do: x, else: alpha * x

  def leaky_relu_bp(x, alpha), do: if x >= 0, do: 1, else: alpha
end  

defmodule Softmax do
  defstruct [name: :softmax]
  
  @behaviour Activations
  
  defimpl Activations do
    @impl Activations
    def activate(softmax, t) do
      e_x = Nx.exp(Nx.subtract(t, Nx.reduce_max(t, axes: [-1], keep_axes: true)))
      Nx.divide(e_x, Nx.sum(e_x, axes: [-1], keep_axes: true))
    end

    @impl Activations
    def gradient(softmax, t) do
      p = Activations.activate(softmax, t)
      Nx.multiply(p, Nx.subtract(1, p))
    end
  end  
end  