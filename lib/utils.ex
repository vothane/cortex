defmodule Utils do
  import Matrex
  alias Matrex
  alias Utils
  
  def sum_of_cols(m) do # equivalent to numpy sum with parameter axis=0 & keepdims=True
    m
    |> transpose
    |> sum_of_rows
    |> transpose
  end
  
  def sum_of_rows(m) do # equivalent to numpy sum with parameter axis=1 & keepdims=True
    m
    |> list_of_rows
    |> Enum.map(&sum/1)
    |> (&(Matrex.new([&1]))).()
    |> transpose
  end  
  
  def add_m_v(mat, vec) do # add matrix to vector: m+v in nummpy
    mat
    |> list_of_rows 
    |> Enum.map(fn row -> [add(row, vec)] end)
    |> Matrex.new
  end  
  
  def subtract_m_v(mat, vec) do # subtract matrix to vector: m-v in nummpy
    mat
    |> list_of_rows 
    |> Enum.map(fn row -> [subtract(row, vec)] end)
    |> Matrex.new
  end
  
  def mult_m_v(mat, vec) do # multiply matrix to vector: m*v in nummpy
    mat
    |> list_of_rows 
    |> Enum.map(fn row -> [multiply(row, vec)] end)
    |> Matrex.new
  end
  
  def clip(m, min, max) do # same as numpy clip
    f = fn x -> Utils.norm(x, min, max) end
    Matrex.apply(m, f)
  end
  
  def norm(x, min, max) do
    cond do
      x < min -> min
      x > max -> max
      true -> x
    end
  end
    
  def max_of_cols(m) do # equivalent to numpy max with parameter axis=0 & keepdims=True
    m
    |> transpose
    |> max_of_rows
    |> transpose
  end
  
  def max_of_rows(m) do # equivalent to numpy max with parameter axis=1 & keepdims=True
    m
    |> list_of_rows
    |> Enum.map(&max/1)
    |> (&(Matrex.new([&1]))).()
    |> transpose
  end
  
  def mean_of_cols(m) do # equivalent to numpy mean with parameter axis=0
    m
    |> transpose
    |> mean_of_rows
  end
  
  def mean_of_rows(m) do # equivalent to numpy mean with parameter axis=1
    num_els = fn ({rows, cols} = _shape) -> rows * cols end
    row_mean = fn (row) -> Matrex.sum(row) / (num_els.(Matrex.size(row))) end
    
    m
    |> list_of_rows
    |> Enum.map(row_mean)
    |> (&(Matrex.new([&1]))).()
  end
  
  def variance_of_cols(m) do # equivalent to numpy var with parameter axis=1
    m
    |> transpose
    |> variance_of_rows
  end
  
  def variance_of_rows(m) do # equivalent to numpy var with parameter axis=1
    means = mean_of_rows(m)
    num_els = fn ({rows, cols} = _shape) -> rows * cols end
    mean_fn = fn (m) -> Matrex.sum(m) / num_els.(Matrex.size(m)) end
    sq_devs = Matrex.apply(m, fn x, row, col -> :math.pow(abs(x - Matrex.at(means, 1, row)), 2) end)
    variances = list_of_rows(sq_devs)
             |> Enum.map(mean_fn)
             |> (&(Matrex.new([&1]))).()
    variances         
  end

  def is_vector?(m) do
    if is_number(m) do
      false
    else  
      {rows, _} = Matrex.size(m)
      rows == 1
    end  
  end  
end