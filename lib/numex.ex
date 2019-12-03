import Utils

defmodule Numex do
  def add(m1, m2) do
    num_vecs = Enum.count([m1, m2], fn m -> is_vector?(m) end)
    
    if Enum.member?([0, 2], num_vecs) or Enum.any?([m1, m2], &is_number/1) do
      Matrex.add(m1, m2)
    else
      vec = List.first(Enum.filter([m1, m2], fn m -> is_vector?(m) end))
      mat = List.first(Enum.reject([m1, m2], fn m -> is_vector?(m) end))
      add_m_v(mat, vec)
    end  
  end
  
  def multiply(m1, m2) do
    num_vecs = Enum.count([m1, m2], fn m -> is_vector?(m) end)
  
    if Enum.member?([0, 2], num_vecs) or Enum.any?([m1, m2], &is_number/1) do
      Matrex.multiply(m1, m2)
    else
      vec = List.first(Enum.filter([m1, m2], fn m -> is_vector?(m) end))
      mat = List.first(Enum.reject([m1, m2], fn m -> is_vector?(m) end))
      mult_m_v(mat, vec)
    end 
  end
end  