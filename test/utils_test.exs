defmodule UtilsTest do
  use ExUnit.Case
  doctest Utils
  
  test "sum of cols ala sum along axis 0" do
    arr = [[14, 17, 12, 33, 44],    
           [15,  6, 27,  8, 19],   
           [23,  2, 54,  1,  4]]   
    m = Matrex.new(arr)
    sum = Utils.sum_of_cols(m)
    assert sum == Matrex.new([[52, 25, 93, 42, 67]])
  end
   
  test "sum of rows ala sum along axis 1" do
    arr = [[14, 17, 12, 33, 44],    
           [15,  6, 27,  8, 19],   
           [23,  2, 54,  1,  4]]   
    m = Matrex.new(arr)
    sum = Utils.sum_of_rows(m)
    assert sum == Matrex.new([[120], [75], [84]])
  end
end  