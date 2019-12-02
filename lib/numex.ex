import Utils

defmodule Numex do
  def add(m1, m2) do
    num_vecs = Enum.count([m1, m2], fn m -> is_vector?(m) end)
    
    if num_vecs == 0 or num_vecs == 2 do
      Matrex.add(m1, m2)
    else
      vec = List.first(Enum.filter([m1, m2], fn m -> is_vector?(m) end))
      mat = List.first(Enum.reject([m1, m2], fn m -> is_vector?(m) end))
      add_m_v(mat, vec)
    end  
  end
  
  def multiply(m1, m2) do
    num_vecs = Enum.count([m1, m2], fn m -> is_vector?(m) end)
  
    if num_vecs == 0 or num_vecs == 2 do
      Matrex.multiply(m1, m2)
    else
      vec = List.first(Enum.filter([m1, m2], fn m -> is_vector?(m) end))
      mat = List.first(Enum.reject([m1, m2], fn m -> is_vector?(m) end))
      mult_m_v(mat, vec)
    end 
  end
end  