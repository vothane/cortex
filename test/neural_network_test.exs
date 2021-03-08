defmodule NeuralNetworkTest do
  use ExUnit.Case
  doctest NeuralNetwork

  import Nx
  import NeuralNetwork
  import SquareLoss
  import CrossEntropy
  import SGD
  import RMSprop
  import Activation
  import TanH
  import Sigmoid
  import Dropout
  import NimbleCSV

  @tag timeout: :infinity
  test "cortex with XOR problem" do
    {status, sgd} = sgd(%{w_: nil, momentum: 0.0, learning_rate: 0.1})
    {status, loss} = square_loss(%{})
    {status, nn} = neural_network(sgd, loss)
    {status, relu_layer} = activation(:relu)
    {status, sigmoid_layer} = activation(:sigmoid)

    NeuralNetwork.add(nn, Dense.dense(%{shape_input: {1,2}, n: 16}))
    NeuralNetwork.add(nn, relu_layer)
    NeuralNetwork.add(nn, Dense.dense(%{n: 1}))
    NeuralNetwork.add(nn, sigmoid_layer)

    training_set = [{Nx.tensor([[0.0, 0.0]]), Nx.tensor([[0.0]])},
                    {Nx.tensor([[0.0, 1.0]]), Nx.tensor([[1.0]])},
                    {Nx.tensor([[1.0, 0.0]]), Nx.tensor([[1.0]])},
                    {Nx.tensor([[1.0, 1.0]]), Nx.tensor([[0.0]])}]

    {x_train, y_train} = Enum.unzip(training_set)
    epochs = 3000

    NeuralNetwork.fit(nn, x_train, y_train, epochs)

    is_within? = fn (y, actual, delta) -> (actual - delta) < y and (actual + delta) > y end

    y_0_0 = NeuralNetwork.forward_propogate(nn, Nx.tensor([[0.0, 0.0]]))
    y_0_1 = NeuralNetwork.forward_propogate(nn, Nx.tensor([[0.0, 1.0]]))
    y_1_0 = NeuralNetwork.forward_propogate(nn, Nx.tensor([[1.0, 0.0]]))
    y_1_1 = NeuralNetwork.forward_propogate(nn, Nx.tensor([[1.0, 1.0]]))

    # tolerance here is generous and indicates that deep learner architecture is not
    # optimal, my guess is that the tanh activation layer shouldn't be used b/c the
    # tanh function is known to have some issues in some cases
    # but now I'm just concern with the DL working correctly not accuracy

    tolerance = 0.25

    # assert is_within?.(Matrex.at(y_0_0, 1, 1), 0, tolerance)
    # assert is_within?.(Matrex.at(y_0_1, 1, 1), 1, tolerance)
    # assert is_within?.(Matrex.at(y_1_0, 1, 1), 1, tolerance)
    # assert is_within?.(Matrex.at(y_1_1, 1, 1), 0, tolerance)
    IO.inspect(y_0_0)
    IO.inspect(y_0_1)
    IO.inspect(y_1_0)
    IO.inspect(y_1_1)
  end

  @tag timeout: :infinity
  test "iris classification" do
    NimbleCSV.define(IrisParser, separator: ",", escape: "\0")

    species_map = %{"setosa" => 0, "versicolor" => 1, "virginica" => 2}

    get_data =
      fn file ->
        file
        |> File.stream!(read_ahead: 160)
        |> IrisParser.parse_stream
        |> Stream.map(
             fn [sepal_len, sepal_width, petal_len, petal_width, species] ->
               x_row = Enum.map([sepal_len, sepal_width, petal_len, petal_width], &String.to_float/1)
               y = Utils.one_hot(Map.get(species_map, species), 3)
               {x_row, Nx.tensor(y)}
             end)
        |> Enum.unzip
      end

    {x_train, y_train} = get_data.("test/data/iris_train.csv")
    x_train = Utils.norm_data_cols(x_train)
    x_train = Enum.map(x_train, fn row -> Nx.tensor(row) end)
    IO.inspect(x_train)

    {status, optimizer} = rmsp(%{})
    {status, loss} = cross_entropy(%{})
    {status, iris_classifier} = neural_network(optimizer, loss)
    {status, activ_layer1} = activation(:relu)
    {status, activ_layer2} = activation(:softmax)

    NeuralNetwork.add(iris_classifier, Dense.dense(%{shape_input: {1,4}, n: 10}))
    NeuralNetwork.add(iris_classifier, activ_layer1)
    NeuralNetwork.add(iris_classifier, Dense.dense(%{n: 10}))
    NeuralNetwork.add(iris_classifier, activ_layer1)
    NeuralNetwork.add(iris_classifier, Dense.dense(%{n: 3}))
    NeuralNetwork.add(iris_classifier, activ_layer2)

    epochs = 500
    
    NeuralNetwork.fit(iris_classifier, x_train, y_train, epochs)
    
    {x_test, y_test} = get_data.("test/data/iris_test.csv")

    y_preds = Enum.map(x_test, fn x -> NeuralNetwork.forward_propogate(iris_classifier, x) end)

    IO.inspect(y_test)
    IO.puts("------------------------------------------------------------------------------------------------------------------")
    IO.inspect(y_preds)
  end


  # @doc """ 
  #   true positives eqv. discrimator correctly ids true image
  #   true negatives eqv. discrimator uncorrectly ids fake made by generator as true image
  #   false positives eqv. discrimator uncorrectly ids true image as fake
  #   false negatives eqv. discrimator correctly ids fake made by generator

  #   Goal is to make true negatives score for generator as high as possible.
  # """
  # @tag timeout: :infinity
  # test "simple GAN from blog.paperspace.com/implementing-gans-in-tensorflow" do

  #   samples = 100
  #   epochs = 1000
  #   latent_dim = 2

  #   sample_data =
  #     fn () ->
  #       get_x = fn () -> (:rand.uniform - 0.5) * 100 end
  #       fx = fn (x) -> x*x end

  #       x_data = Enum.map(1..samples, fn (_) -> get_x.() end)
  #       data = Enum.map(x_data, fn (x) -> Matrex.new([[x, fx.(x)]]) end)
  #       data
  #     end

  #   latent_noise =
  #     fn () ->
  #       f = fn () -> -1 + (:rand.uniform() * 2) end
  #       Enum.map(1..samples, fn (_) -> Matrex.new(1, latent_dim, fn () -> f.() end) end)
  #     end

  #   {status, rmsp1} = rmsp(%{})
  #   {status, loss1} = cross_entropy(%{})
  #   {status, generator} = neural_network(rmsp1, loss1)
  #   {status, lrelu_layer} = activation(:leaky_relu)

  #   NeuralNetwork.add(generator, Dense.dense(%{shape_input: {1,latent_dim}, n: 16}))
  #   NeuralNetwork.add(generator, lrelu_layer)
  #   NeuralNetwork.add(generator, Dense.dense(%{n: 16}))
  #   NeuralNetwork.add(generator, lrelu_layer)
  #   NeuralNetwork.add(generator, Dense.dense(%{n: 2}))

  #   {status, rmsp2} = rmsp(%{})
  #   {status, loss2} = cross_entropy(%{})
  #   {status, discriminator} = neural_network(rmsp2, loss2)
  #   {status, sigmoid_layer} = activation(:sigmoid)

  #   NeuralNetwork.add(discriminator, Dense.dense(%{shape_input: {1,2}, n: 16}))
  #   NeuralNetwork.add(discriminator, lrelu_layer)
  #   NeuralNetwork.add(discriminator, Dense.dense(%{n: 16}))
  #   NeuralNetwork.add(discriminator, lrelu_layer)
  #   NeuralNetwork.add(discriminator, Dense.dense(%{n: 1}))
  #   NeuralNetwork.add(discriminator, sigmoid_layer)

  #   real_data = sample_data.()
  #   fake_data = Enum.map(latent_noise.(), fn x -> NeuralNetwork.forward_propogate(generator, x) end)
    
  #   valid = Stream.repeatedly(fn -> Matrex.new([[1]]) end) |> Enum.take(samples)
  #   invalid = Stream.repeatedly(fn -> Matrex.new([[0]]) end) |> Enum.take(samples)
    
  #   NeuralNetwork.fit(discriminator, real_data, valid, epochs)
  #   NeuralNetwork.fit(discriminator, fake_data, invalid, epochs)

  #   {status, rmsp3} = rmsp(%{})
  #   {status, loss3} = cross_entropy(%{})
  #   {status, combined} = neural_network(rmsp3, loss3)

  #   NeuralNetwork.set_trainable(discriminator, false)
  #   disclays = NeuralNetwork.get(discriminator, :layers)
  #   f = fn l -> %mod{} = Agent.get(l, &(&1)); mod end
  #   train_flags = Enum.map(disclays, &(f.(&1).get(&1, :trainable)))
  #   assert Enum.all?(train_flags, fn trainable -> trainable == false end)

  #   NeuralNetwork.put(combined, :layers, NeuralNetwork.get(generator, :layers) ++ NeuralNetwork.get(discriminator, :layers))
  #   NeuralNetwork.fit(combined, latent_noise.(), valid, epochs)

  #   untrained_gen = generator
  #   combined_layers = NeuralNetwork.get(combined, :layers)
  #   last_idx_gen = Enum.count(NeuralNetwork.get(generator, :layers)) - 1
  #   NeuralNetwork.put(generator, :layers, Enum.slice(combined_layers, 0..last_idx_gen))

  #   accuracy_score =
  #     fn (y_true, y_pred) ->
  #       score = MapSet.intersection(MapSet.new(y_true), MapSet.new(y_pred)) |> MapSet.size
  #       total = Enum.count(y_true)
  #       score/total
  #     end 

  #   true_positives = sample_data.()
  #   y_preds_tp = Enum.map(true_positives, fn (img) -> NeuralNetwork.forward_propogate(discriminator, img) end)

  #   assert Enum.all?(y_preds_tp, fn pred -> pred == [1] end)

  #   true_negatives = Enum.map(latent_noise.(), fn x -> NeuralNetwork.forward_propogate(generator, x) end)
  #   y_preds_tn = Enum.map(true_negatives, fn (img) -> NeuralNetwork.forward_propogate(discriminator, img) end)

  #   assert Enum.all?(y_preds_tn, fn pred -> pred == [1] end)

  # end
end
