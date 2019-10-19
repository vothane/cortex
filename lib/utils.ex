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
end