## Backpropagation Implementation Helper Formulas

#### Notations

- $b^l_j$ for the bias of the $j^{th}$ neuron in the $l^{th}$ layer
- $w^l_{jk}$ for the weight for the connection from the $k^{th}$ neuron in the
  $(l−1)^{th}$ layer to the $j^{th}$ neuron in the $l^{th}$ layer
- $z^l_j$ for the weighted input to the activation function for neuron $j$ in layer $l$
- $a^{l}_{j}$ for the activation of the $j^{th}$ neuron in the $l^{th}$ layer
- $\sigma$ for the sigmoid function

#### Formulas

1. the activation $a^{l}_{j}$ of the $j^{th}$ neuron in the $l^{th}$ layer is related to the activations in the $(l−1)^{th}$ layer
   $$
   \begin{eqnarray} 
     a^{l}_j = \sigma\left( \sum_k w^{l}_{jk} a^{l-1}_k + b^l_j \right)
   \end{eqnarray}
   $$
   which is equivalent to
   $$
   \begin{eqnarray} 
     a^{l} = \sigma(w^l a^{l-1}+b^l)
   \end{eqnarray}
   $$

2. the cost function 
   $$
   \begin{eqnarray}
     C = \frac{1}{2} \|y-a^L\|^2 = \frac{1}{2} \sum_j (y_j-a^L_j)^2
   \end{eqnarray}
   $$

3. the gradient $δ^l_j$ of neuron $j$ in layer $l$
   $$
   \begin{eqnarray} 
     \delta^l_j = \frac{\partial C}{\partial z^l_j}
   \end{eqnarray}
   $$

4. the gradient in the output layer, $δ^L$
   $$
   \begin{eqnarray} 
     \delta^L_j = \frac{\partial C}{\partial z^L_j}= \frac{\partial C}{\partial a^L_j} \sigma'(z^L_j) = (a_j^L-y_j)\sigma'(z^L_j)
   \end{eqnarray}
   $$
   which is equivalent to
   $$
   \begin{eqnarray} 
     \delta^L = (a^L-y)\odot\sigma'(z^L)
   \end{eqnarray}
   $$

5. the gradient $δ^l$ in terms of the gradient in the next layer, $δ^{l+1}$, where $l < L$
   $$
   \begin{eqnarray} 
     \delta^l = ((w^{l+1})^T \delta^{l+1}) \odot \sigma'(z^l)
   \end{eqnarray}
   $$
   *Proof*
   $$
   \begin{eqnarray}
     \delta^l_j & = & \frac{\partial C}{\partial z^l_j} \\
     & = & \sum_k \frac{\partial C}{\partial z^{l+1}_k} \frac{\partial z^{l+1}_k}{\partial z^l_j} \\ 
     & = & \sum_k \frac{\partial z^{l+1}_k}{\partial z^l_j} \delta^{l+1}_k
   \end{eqnarray}
   $$
   where 
   $$
   \begin{eqnarray}
     z^{l+1}_k = \sum_j w^{l+1}_{kj} a^l_j +b^{l+1}_k = \sum_j w^{l+1}_{kj} \sigma(z^l_j) +b^{l+1}_k
   \end{eqnarray}
   $$
   so
   $$
   \begin{eqnarray}
     \frac{\partial z^{l+1}_k}{\partial z^l_j} = w^{l+1}_{kj} \sigma'(z^l_j)
   \end{eqnarray}
   $$
   then we have
   $$
   \begin{eqnarray}
     \delta^l_j = \sum_k w^{l+1}_{kj}  \delta^{l+1}_k \sigma'(z^l_j)
   \end{eqnarray}
   $$
   ​                                           															$\square$

6. the gradient of the bias $b^l$ in layer $l$ 
   $$
   \begin{eqnarray}  \frac{\partial C}{\partial b^l_j} = \frac{\partial C}{\partial z^l_j}\frac{\partial z^l_j}{\partial b^l_j}=\frac{\partial C}{\partial z^l_j}\frac{\partial (w^{l}a^{l-1} + b^l_j)}{\partial b^l_j}
     =\delta^l_j
   \end{eqnarray}
   $$
   which is equivalent to
   $$
   \begin{eqnarray}  \frac{\partial C}{\partial b^l}=\delta^l
   \end{eqnarray}
   $$

7. the gradient of the weight $w^l$ in layer $l$
   $$
   \begin{eqnarray}
     
     \frac{\partial C}{\partial w^l_{jk}} = a^{l-1}_k \delta^l_j
   \end{eqnarray}
   $$
   which is equivalent to
   $$
   \begin{eqnarray}
     
     \frac{\partial C}{\partial w^l} = a^{l-1}\delta^l
   \end{eqnarray}
   $$





