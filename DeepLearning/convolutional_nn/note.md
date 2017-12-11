## Week 1

#### Edge detection

Sobel filter: ${\begin{bmatrix}1&0&-1\\2&0&-2\\1&0&-1\end{bmatrix}}$

Scharr filter: ${\begin{bmatrix}3&0&-3\\10&0&-10\\3&0&-3\end{bmatrix}}$

Arbitrary filter: ${\begin{bmatrix}w_1&w_2&w_3\\w_4&w_5&w_6\\w_7&w_8&w_9\end{bmatrix}}$ with 9 parameters to learn.

#### Padding

*Valid* Convolution: ${n \times n * f \times f \rightarrow (n-f+1) \times (n-f+1)}$

*Same* convolution: ${n+2p-f+1 = n \rightarrow p = (f-1)/2}$

$f$ is usually odd.

#### Strided convolution

With padding $p$ and stride $s$:
${n \times n * f \times f \rightarrow \cfrac{n+2p-f+1}{s} \times \cfrac{n+2p-f+1}{s}}$

#### Convolution over volume

e.g. ${6 \times 6 \times 3 * 3 \times 3 \times 3 \rightarrow 4 \times 4}$

#### Multiple filters

Summary: ${n \times n \times n_c * f \times f \times n_c \rightarrow (n-f+1) \times (n-f+1) \times n_c^\prime}$

$n^\prime$ is the number of filters used in the convolution.

#### Number of parameters

If you have $10$ filters that are $3\times3\times3$ in one layer of a neural network, how many parameters are there in this layer?

$(3 \times 3 \times 3 + 1) \times 10 = 280$

There is one bias parameter in each filter.

#### Summary of notation

If layer $l$ is a convolution layer:

${f^{[l]}}$ = filter size <br>
${p^{[l]}}$ = padding <br>
${s^{[l]}}$ = stride <br>
${n_c^{[l]}}$ = number of filters

Input: <br>${n_H^{[l-1]} \times n_W^{[l-1]} \times n_c^{[l-1]}}$

Output: <br>${n_H^{[l]} \times n_W^{[l]} \times n_c^{[l]}}$

where: <br>${n_H^{[l]} = \lfloor \cfrac{n_H^{[l-1]} + 2p^{[l]}-f^{[l]}}{s^{[l]}}+1 \rfloor}$
<br>${n_W^{[l]} = \lfloor \cfrac{n_W^{[l-1]} + 2p^{[l]}-f^{[l]}}{s^{[l]}}+1 \rfloor}$

Each filter is: ${f^{[l]} \times f^{[l]} \times n_c^{[l-1]}}$ <br>
Activations: ${a^{[l]} \rightarrow n_H^{[l]} \times n_W^{[l]} \times n_c^{[l]}}$ <br>
Weights: ${f^{[l]} \times f^{[l]} \times n_c^{[l-1]} \times n_c^{[l]}}$ <br>
Bias: ${1 \times 1 \times 1 \times n_c^{[l]}}$

#### Types of layer in a convolutional network
- convolution $(CV)$<br>
- Pooling $(POOL)$<br>
- Fully connected $(FC)$

#### Pooling

Hyperparameters: $f, s, type$ <br>
The formula ${\cfrac{n+2p-f+1}{s}}$ also works for pooling.

#### Example
LeNet

#### Conclusion

**Parameter sharing**: a feature detector that's useful in one part of the image is probably useful in another part of the image.

**Sparsity of connections**: In each layer, each output value depends on a small number of inputs.
