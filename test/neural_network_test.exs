defmodule NeuralNetworkTest do
  use ExUnit.Case
  doctest NeuralNetwork

  import Matrex
  import NeuralNetwork
  import CrossEntropy
  import SGD
  import RMSprop
  import Activation
  import TanH
  import Sigmoid

  test "cortex with XOR problem without loss function" do
    {status, sgd} = sgd(%{w_: nil, momentum: 0.0, learning_rate: 0.1})
    {status, nn} = neural_network(sgd)
    {status, tanh_layer} = activation(:tanh)
    {status, sigmoid_layer} = activation(:sigmoid)

    NeuralNetwork.add(nn, Dense.dense(%{shape_input: {1,2}, n: 2}))
    NeuralNetwork.add(nn, tanh_layer)
    NeuralNetwork.add(nn, Dense.dense(%{n: 1}))
    NeuralNetwork.add(nn, sigmoid_layer)

    training_set = [{Matrex.new([[0.0, 0.0]]), Matrex.new([[0.0]])},
                    {Matrex.new([[0.0, 1.0]]), Matrex.new([[1.0]])},
                    {Matrex.new([[1.0, 0.0]]), Matrex.new([[1.0]])},
                    {Matrex.new([[1.0, 1.0]]), Matrex.new([[0.0]])}]

    Enum.reduce(1..5000, [], fn(_, _) ->
      Enum.reduce(training_set, [], fn({x, y}, _) ->
        outputs = NeuralNetwork.forward_propogate(nn, x)
        output_deltas = Matrex.apply(outputs, fn y_output -> y_output * (1 - y_output) * (y_output - Matrex.at(y, 1, 1)) end)
        NeuralNetwork.backward_propogate(nn, output_deltas)
      end)
    end)

    is_within? = fn (y, actual, delta) -> (actual - delta) < y and (actual + delta) > y end

    y_0_0 = NeuralNetwork.forward_propogate(nn, Matrex.new([[0.0, 0.0]]))
    y_0_1 = NeuralNetwork.forward_propogate(nn, Matrex.new([[0.0, 1.0]]))
    y_1_0 = NeuralNetwork.forward_propogate(nn, Matrex.new([[1.0, 0.0]]))
    y_1_1 = NeuralNetwork.forward_propogate(nn, Matrex.new([[1.0, 1.0]]))

    # tolerance here is generous and indicates that deep learner architecture is not
    # optimal, my guess is that the tanh activation layer shouldn't be used b/c the
    # tanh function is known to have some issues in some cases
    # but now I'm just concern with the DL working correctly not accuracy

    tolerance = 0.25

    assert is_within?.(Matrex.at(y_0_0, 1, 1), 0, tolerance)
    assert is_within?.(Matrex.at(y_0_1, 1, 1), 1, tolerance)
    assert is_within?.(Matrex.at(y_1_0, 1, 1), 1, tolerance)
    assert is_within?.(Matrex.at(y_1_1, 1, 1), 0, tolerance)
  end
  

  @doc """ 
    true positives eqv. discrimator correctly ids true image
    true negatives eqv. discrimator uncorrectly ids fake made by generator as true image
    false positives eqv. discrimator uncorrectly ids true image as fake
    false negatives eqv. discrimator correctly ids fake made by generator

    Goal is to make true negatives score for generator as high as possible.
  """
  @tag timeout: :infinity
  test "simple GAN from blog.paperspace.com/implementing-gans-in-tensorflow" do
    latent_dim = 16
    samples = 400
    scale = 100

    get_x = fn () -> scale * :rand.uniform - 0.5 end
    fx = fn (x) -> 10 + x*x end

    sample_data =
      fn () ->
        x_data = Enum.map(1..samples, fn (_) -> get_x.() end)
        data = Enum.map(x_data, fn (x) -> Matrex.new([[x, fx.(x)]]) end)
        data
      end

    sample_noise =
      fn () ->
        f = fn () -> :rand.normal(-1, 1) end
        noise = Enum.map(1..samples, fn (_) -> Matrex.new([[f.(), f.()]]) end)
        noise
      end

    {status, rmsp1} = rmsp(%{})
    {status, loss1} = cross_entropy(%{})
    {status, generator} = neural_network(rmsp1, loss1)
    {status, lrelu_layer} = activation(:leaky_relu)
    {status, tanh_layer} = activation(:tanh)

    NeuralNetwork.add(generator, Dense.dense(%{shape_input: {1,2}, n: latent_dim}))
    NeuralNetwork.add(generator, lrelu_layer)
    NeuralNetwork.add(generator, Dense.dense(%{n: 2}))
    NeuralNetwork.add(generator, tanh_layer)

    {status, rmsp2} = rmsp(%{})
    {status, loss2} = cross_entropy(%{})
    {status, discriminator} = neural_network(rmsp2, loss2)
    {status, lrelu_layer} = activation(:leaky_relu)
    {status, softmax_layer} = activation(:softmax)

    NeuralNetwork.add(discriminator, Dense.dense(%{shape_input: {1,2}, n: latent_dim}))
    NeuralNetwork.add(discriminator, lrelu_layer)
    NeuralNetwork.add(discriminator, Dense.dense(%{n: 2}))
    NeuralNetwork.add(discriminator, softmax_layer)

    imgs = sample_data.()
    noise = sample_noise.()
    gen_imgs = Enum.map(noise, fn x -> NeuralNetwork.forward_propogate(generator, x) end)  
    
    valid = Stream.repeatedly(fn -> Matrex.new([[1, 0]]) end) |> Enum.take(samples)
    fake = Stream.repeatedly(fn -> Matrex.new([[0, 1]]) end) |> Enum.take(samples)
    
    NeuralNetwork.fit(discriminator, imgs, valid, samples)
    NeuralNetwork.fit(discriminator, gen_imgs, fake, samples)

    {status, rmsp3} = rmsp(%{})
    {status, loss3} = cross_entropy(%{})
    {status, combined} = neural_network(rmsp3, loss3)

    NeuralNetwork.set_trainable(discriminator, false)
    disclays = NeuralNetwork.get(discriminator, :layers)
    f = fn l -> %mod{} = Agent.get(l, &(&1)); mod end
    train_flags = Enum.map(disclays, &(f.(&1).get(&1, :trainable)))
    assert Enum.all?(train_flags, fn trainable -> trainable == false end)

    NeuralNetwork.put(combined, :layers, NeuralNetwork.get(generator, :layers) ++ NeuralNetwork.get(discriminator, :layers))

    NeuralNetwork.fit(combined, noise, valid, samples)

    untrained_gen = generator
    combined_layers = NeuralNetwork.get(combined, :layers)
    last_idx_gen = Enum.count(NeuralNetwork.get(generator, :layers)) - 1
    NeuralNetwork.put(generator, :layers, Enum.slice(combined_layers, 0..last_idx_gen))

    accuracy_score =
      fn (y_true, y_pred) ->
        score = MapSet.intersection(MapSet.new(y_true), MapSet.new(y_pred)) |> MapSet.size
        total = Enum.count(y_true)
        score/total
      end 

    true_positives = sample_data.()
    y_preds_tp = Enum.map(true_positives, fn (img) -> NeuralNetwork.forward_propogate(discriminator, img) end)

    IO.puts("____________________________ true positives ____________________________ ")
    IO.inspect(true_positives)
    IO.inspect(y_preds_tp)
    #assert Enum.all?(y_preds, fn pred -> pred == [1, 0] end)

    noise_tn = sample_noise.()
    true_negatives = Enum.map(noise_tn, fn x -> NeuralNetwork.forward_propogate(generator, x) end)
    y_preds_tn = Enum.map(true_negatives, fn (img) -> NeuralNetwork.forward_propogate(discriminator, img) end)

    IO.puts("____________________________ true negatives ____________________________ ")
    IO.inspect(true_negatives)
    IO.inspect(y_preds_tn)
    #assert Enum.all?(y_preds, fn pred -> pred == [1, 0] end)

    IO.puts("____________________________ false negatives ____________________________ ")
    noise_fn = sample_noise.()
    false_negatives = Enum.map(noise_tn, fn x -> NeuralNetwork.forward_propogate(untrained_gen, x) end)
    y_preds_fn = Enum.map(noise_fn, fn (img) -> NeuralNetwork.forward_propogate(discriminator, img) end)
    IO.inspect(false_negatives)
    IO.inspect(y_preds_fn)
  end
end
