import Utils

defmodule Numex do
  def add(m1, m2) do
    calculate(m1,m2, &Matrex.add/2, &add_m_v/2) 
  end

  def subtract(m1, m2) do
    calculate(m1,m2, &Matrex.subtract/2, &subtract_m_v/2)
  end
  
  def multiply(m1, m2) do
    calculate(m1,m2, &Matrex.multiply/2, &mult_m_v/2)
  end
  
  defp calculate(m1, m2, f, g) do
    num_vecs = Enum.count([m1, m2], fn m -> is_vector?(m) end)
  
    if Enum.member?([0, 2], num_vecs) or Enum.any?([m1, m2], &is_number/1) do
      f.(m1, m2)
    else
      vec = List.first(Enum.filter([m1, m2], fn m -> is_vector?(m) end))
      mat = List.first(Enum.reject([m1, m2], fn m -> is_vector?(m) end))
      g.(mat, vec)
    end 
  end
end  