defmodule Utils do
  import Matrex
  alias Matrex
  alias Utils
  
  def sum_of_cols(m) do # equivalent to to numpy sum with parameter axis=0 and keepdimension is True
    sum = m
       |> transpose
       |> sum_of_rows
       |> transpose
  end
  
  def sum_of_rows(m) do # equivalent to to numpy sum with parameter axis=1 and keepdimension is True
    sum = m
       |> list_of_rows
       |> Enum.map(&sum/1)
       |> (&(Matrex.new([&1]))).()
       |> transpose
  end  
end