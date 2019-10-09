defmodule Utils do
  import Matrex
  alias Matrex
  alias Utils
  
  def sum_of_cols(m) do # equivalent to numpy sum with parameter axis=0 & keepdims=True
    sum = m
       |> transpose
       |> sum_of_rows
       |> transpose
  end
  
  def sum_of_rows(m) do # equivalent to numpy sum with parameter axis=1 & keepdims=True
    sum = m
       |> list_of_rows
       |> Enum.map(&sum/1)
       |> (&(Matrex.new([&1]))).()
       |> transpose
  end  
end