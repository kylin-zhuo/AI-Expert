The notes are from the Coursera course of Geoffrey Hinton.

### Networks

Feed-forward neural networks: They compute a series of transformations that change the similarities between cases. The activities of the neurons in each layer are a non-linear function of the activities in the layer below.

Recurrent neural networks: difficult to train; more biologically realistic; very natural way to model sequential data; have the ability to remember information in their hidden state for a long time.

### Perceptron
We can avoid having to figure out a
separate learning rule for the bias by
using a trick:
a) A bias is exactly equivalent to a
weight on an extra input line that
always has an activity of 1. b) We can now learn a bias as if it were a weight.

**Convergence Procedure**: If the output unit is correct, leave its weights alone.
– If the output unit incorrectly outputs a zero, add the input vector to the weight vector.– If the output unit incorrectly outputs a 1, subtract the input vector from the weight vector.
