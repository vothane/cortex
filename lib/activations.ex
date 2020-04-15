defmodule Activations do
  @callback activate(struct, any) :: any
  @callback gradient(struct, any) :: any
end

defprotocol Activations do
  def activate(activation, m)
  def gradient(activation, m)
end
  
defmodule Sigmoid do
  defstruct [name: :sigmoid]
  
  @behaviour Activations
  
  defimpl Activations do
    @impl Activations
    def activate(sigmoid, m) do
      Matrex.apply(m, &Sigmoid.sigmoid/1)
    end
  
    @impl Activations
    def gradient(sigmoid, m) do # derivative of sigmoid
      dfx = &(Sigmoid.sigmoid(&1) * (1.0 - Sigmoid.sigmoid(&1)))
      Matrex.apply(m, dfx)
    end    
  end
  
  def sigmoid(x), do: 1.0 / (1.0 + :math.exp(-x)) 
end

defmodule TanH do
  defstruct [name: :tanh]
  
  @behaviour Activations
  
  defimpl Activations do
    @impl Activations
    def activate(tanh, m) do
      Matrex.apply(m, &TanH.tanh/1)
    end
  
    @impl Activations
    def gradient(tanh, m) do # derivative of tanh
      dfx = &(1 - :math.pow(TanH.tanh(&1), 2))
      Matrex.apply(m, dfx)
    end
  end
  
  def tanh(x), do: :math.tanh(x)
end  

defmodule ReLU do
  defstruct [name: :relu]
  
  @behaviour Activations
  
  defimpl Activations do
    @impl Activations
    def activate(relu, m) do
      Matrex.apply(m, &ReLU.relu/1)
    end

    @impl Activations
    def gradient(relu, m) do
      Matrex.apply(m, &(if &1 >= 0, do: 1, else: 0))
    end
  end
  
  def relu(x), do: if x >= 0, do: x, else: 0
end  

defmodule LeakyReLU do
  defstruct [name: :leaky_relu, alpha: 0.2]
  
  @behaviour Activations
  
  defimpl Activations do
    @impl Activations
    def activate(lrelu, m) do
      Matrex.apply(m, fn x -> LeakyReLU.leaky_relu_fp(x, lrelu.alpha) end)
    end

    @impl Activations
    def gradient(lrelu, m) do
      Matrex.apply(m, fn x -> LeakyReLU.leaky_relu_bp(x, lrelu.alpha) end)
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
    def activate(softmax, m) do
      maxima = Utils.max_of_rows(m)
      f = fn ({row, max}) -> Matrex.subtract(row, Matrex.scalar(max)) end
      diffs = Enum.map(Enum.zip(Matrex.list_of_rows(m), Matrex.list_of_rows(maxima)), f) 
      e_x = Matrex.apply(Matrex.new([diffs]), :exp) 
      sums = Utils.sum_of_rows(e_x)
      g = fn ({row, sum}) -> Matrex.divide(row, Matrex.scalar(sum)) end
      result = Enum.map(Enum.zip(Matrex.list_of_rows(e_x), Matrex.list_of_rows(sums)), g) 
      Matrex.new([result])
    end

    @impl Activations
    def gradient(softmax, m) do
      p = Activations.activate(softmax, m)
      Matrex.multiply(p, Matrex.subtract(1, p))
    end
  end  
end  