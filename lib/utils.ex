defmodule Utils do
  import Nx
  alias Utils
  
  def new_tensor(dim, f) do
    sigma = Enum.reduce(Tuple.to_list(dim), 1, fn x, acc -> x * acc end)
    flat_matrix = Enum.map(1..sigma, fn _ -> f.() end)
    t = Nx.tensor(flat_matrix)
    Nx.reshape(t, dim)
  end
  
  def zeros(dim) do
    sigma = Enum.reduce(Tuple.to_list(dim), 1, fn x, acc -> x * acc end)
    flat_matrix = Enum.map(1..sigma, fn _ -> 0.0 end)
    t = Nx.tensor(flat_matrix)
    Nx.reshape(t, dim)
  end

  def norm(x, min, max) do
    cond do
      x < min -> min
      x > max -> max
      true -> x
    end
  end
    
  # def max_of_cols(m) do # equivalent to numpy max with parameter axis=0 & keepdims=True
  #   m
  #   |> transpose
  #   |> max_of_rows
  #   |> transpose
  # end
  
  # def max_of_rows(m) do # equivalent to numpy max with parameter axis=1 & keepdims=True
  #   m
  #   |> list_of_rows
  #   |> Enum.map(&max/1)
  #   |> (&(Matrex.new([&1]))).()
  #   |> transpose
  # end
  
  # def mean_of_cols(m) do # equivalent to numpy mean with parameter axis=0
  #   m
  #   |> transpose
  #   |> mean_of_rows
  # end
  
  # def mean_of_rows(m) do # equivalent to numpy mean with parameter axis=1
  #   num_els = fn ({rows, cols} = _shape) -> rows * cols end
  #   row_mean = fn (row) -> Matrex.sum(row) / (num_els.(Matrex.size(row))) end
    
  #   m
  #   |> list_of_rows
  #   |> Enum.map(row_mean)
  #   |> (&(Matrex.new([&1]))).()
  # end
  
  # def variance_of_cols(m) do # equivalent to numpy var with parameter axis=1
  #   m
  #   |> transpose
  #   |> variance_of_rows
  # end
  
  # def variance_of_rows(m) do # equivalent to numpy var with parameter axis=1
  #   means = mean_of_rows(m)
  #   num_els = fn ({rows, cols} = _shape) -> rows * cols end
  #   mean_fn = fn (m) -> Matrex.sum(m) / num_els.(Matrex.size(m)) end
  #   sq_devs = Matrex.apply(m, fn x, row, col -> :math.pow(abs(x - Matrex.at(means, 1, row)), 2) end)
  #   variances = list_of_rows(sq_devs)
  #            |> Enum.map(mean_fn)
  #            |> (&(Matrex.new([&1]))).()
  #   variances         
  # end

  def is_vector?(m) do
    if is_number(m) do
      false
    else  
      {rows, _} = Matrex.size(m)
      rows == 1
    end  
  end

  def one_hot(categorical_val, num_categories) do
    hot_code = Enum.map(1..num_categories, fn _ -> 0 end)
    hot_code = List.replace_at(hot_code, categorical_val-1, 1) 
    Nx.tensor(hot_code)
  end  
end