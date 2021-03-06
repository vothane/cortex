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
   
  test "addition of matrix with vector" do
    arr = [[1, 1, 1, 1, 1],    
           [2, 2, 2, 2, 2],   
           [3, 3, 3, 3, 3]]   
    m = Matrex.new(arr)
    v = Matrex.new([[1, 1, 1, 1, 1]])
    sum = Utils.add_m_v(m, v)
    assert sum == Matrex.new([[2, 2, 2, 2, 2],   
                              [3, 3, 3, 3, 3],
                              [4, 4, 4, 4, 4]])
  end
  
  test "clipping of probabilities like numpy" do
    m1 = Utils.clip(Matrex.new([[0.5, 0.5]]), 1.0e-15, 1-1.0e-15)
    m2 = Utils.clip(Matrex.new([[-0.1, 1.1]]), 1.0e-15, 1-1.0e-15)
    assert m1 == Matrex.new([[0.5, 0.5]])
    assert m2 == Matrex.new([[1.0e-15, 1-1.0e-15]])   
  end

  test "one-hot encoding" do
    assert Utils.one_hot(0, 4) == Matrex.new([[1.0, 0.0, 0.0, 0.0]])
    assert Utils.one_hot(1, 4) == Matrex.new([[0.0, 1.0, 0.0, 0.0]])
    assert Utils.one_hot(2, 4) == Matrex.new([[0.0, 0.0, 1.0, 0.0]])
    assert Utils.one_hot(3, 4) == Matrex.new([[0.0, 0.0, 0.0, 1.0]])
  end
end  