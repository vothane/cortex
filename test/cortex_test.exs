defmodule CortexTest do
  use ExUnit.Case
  doctest Cortex

  test "greets the world" do
    assert Cortex.hello() == :world
  end
end
