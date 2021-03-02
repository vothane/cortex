# Cortex

__Cortex__ does not aim to bring practical, extensible, productive, and performant deep learning to Elixir. This for learning purposes only, ___not___ for ~~production~~.


### IN PROGRESS  
___Recurrent Neural Networks___
___Generative Adversarial Networks___
![](https://66.media.tumblr.com/e36ab29c9357a7bc309fdc5971409aa7/tumblr_okoovm5sRD1rzu2xzo4_r1_400.gif)

### PLANNED

___Convolutional Neural Networks___
![](https://i.imgur.com/yhPAgPK.gif)
  
### Issues and ToDo
![](https://user-images.githubusercontent.com/256203/70104032-bb050400-1634-11ea-8469-7d48f8ae1c46.gif)

* find out cross entropy loss only works for two classes. fails for iris classifier since it has three
* weights in XOR network sometimes do not converge and failed test. Use Xavier initialization.
* Tighten type inference scoping
* softmax blows up since Nx.exp numerically overflows on some numbers such as 725.6691854756205.
easy fix is to make floats 128 bit precision which IDk if erlang/elixir supports. Won't be asking
the project people at Nx bc they probably hate me. Since I ratted out two of the main Nx contributors
on this issue https://github.com/versilov/matrex/issues/20 - one directly, the other indirectly.
No apologies my dudes. Play stupid games, get stupid prizes. Especially if that prize is a whirlwind.