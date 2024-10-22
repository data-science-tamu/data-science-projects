# Summary of things to do.
## Wednesday October 16th 2024
#### The Model
- `init_weight_and_bias`: Of soft-plus doesn't fix nan issue,
try with higher (0.1?) initial mean for `trunc_normal_`
- `FittingModel`: Add positional embedding layer (to increase dimensionality
of the input). Try this after converting displacement model to use strain.
- RE-OCCURRING: Find weights that work for the model.
    - In so doing, save more data on loss values for each weight test.
    - Graph weights on the same plot to see how the optimizer is doing it.
      For example, is it prioritizing when of the loss terms over another? And
      do we want it to prioritize that loss term.
    - loss_e shouldn't be very important (just making sure e isn't 0), if it goes to 0 increase WEIGHT_E
    - pde loss should be most important

#### Animations
- Add animation for loss values (might help with seeing where the model
currently is in the animation).
- Change relative error to abs difference (Remove high error values).
- See why the color bar gradient isn't changing.
- Possibly display the gradients of the displacement fit (the main cause
of the problem, because it amplifies the blurriness of the image.)

#### UN-FINISHED Previous Tasks:
- Model: Try running with displacement fit model fixed (not part of loss during
  training stage).