# Higgs mass constraint equation derivation
The unknown longitudinal momentum of neutrino from W1 decay can be solved by applying the Higgs mass constraint, giving the NN-regressed transverse momenta of the neutrinos and full info of W0.
Set the Higgs mass constraint equation as below:
$$
\begin{align*}
        m_h^2
        &= E_h^2 - p_h^2 = (E_{W_0} + E_{W_1})^2
         - \|\vec{p}_{W_0} + \vec{p}_{W_1}\|^2 \\
        &= (E_{W_0} + E_{\ell_1} + \sqrt{p^2_{x\nu_1} + p^2_{y\nu_1} + p^2_{z\nu_1}})^2  \\
        &\quad - (p_{xh} + p_{yh} + p_{zW_0} + p_{x\ell_1} + p_{x\ell_1})^2
\end{align*}
$$

$$
\begin{equation*}
    \begin{cases}
        &A \equiv E_{W_0} + E_{W_1} \\
        &B \equiv p_{xh} + p_{yh} + p_{zW_0} + p_{x\ell_1} \\
        &p_{T\nu_1} \equiv \sqrt{p^2_{x\nu_1} + p^2_{y\nu_1}}
    \end{cases}
\end{equation*}
$$

$$
\begin{align*}
    m_h^2
    &= \left(A^2
     + p_{T\nu_1}^2
     + p_{z\nu_1}^2
     + 2A\sqrt{p_{T\nu_1}^2 + p_{z\nu_1}^2}\right)
     - \left(B^2 + p_{z\nu_1}^2 + 2B p_{z\nu_1}\right)
\end{align*}
$$

$$
\begin{align*}
        m_h^2
        &= (A^2 - B^2)
         + p_{T\nu_1}^2
         + 2\left(
             A\sqrt{p_{T\nu_1}^2 + p_{z\nu_1}^2}
             - B p_{z\nu_1}
           \right)
\end{align*}
$$

$$
\begin{align*}
    \frac{1}{4}
    \left( m_h^2 + (B^2 - A^2) - p_{T\nu_1}^2 \right)^2
    &= \left(A\sqrt{p_{T\nu_1}^2 + p_{z\nu_1}^2} - B p_{z\nu_1} \right)^2 \\
    &= A^2(p_{T\nu_1}^2 + p_{z\nu_1}^2) + B^2 p^2_{z\nu_1} - 2ABp_{z\nu_1}\sqrt{p_{T\nu_1}^2 + p_{z\nu_1}^2} \\
    &= A^2p_{T\nu_1}^2 + (A^2 + B^2) p_{z\nu_1}^2 - 2AB\sqrt{p_{z\nu_1}^2(p_{T\nu_1}^2 + p_{z\nu_1}^2)}
\end{align*}
$$

$$
\begin{align*}
    C \equiv \frac{1}{4}
    \left[ m_h^2 + (B^2 - A^2) - p_{T\nu_1}^2 \right]^2 - A^2p_{T\nu_1}^2
\end{align*}
$$

$$
\begin{align*}
    \left[ C - (A^2 + B^2) p_{z\nu_1}^2 \right]^2 &= 4A^2B^2 p_{z\nu_1}^2(p_{T\nu_1}^2 + p_{z\nu_1}^2), \\
    C^2 + (A^2 + B^2)^2 p_{z\nu_1}^4 -2C (A^2 + B^2) p_{z\nu_1}^2&= 4A^2B^2p_{T\nu_1}^2p_{z\nu_1}^2 +  4A^2B^2p_{z\nu_1}^4
\end{align*}
$$

$$
\begin{align}
    &p_{z\nu_1}^4 (4A^2B^2 - A^4 - B^4 -2A^2B^2) + p_{z\nu_1}^2 \left[4A^2B^2p_{T\nu_1}^2 + 2C(A^2 + B^2)\right] -C^2 =0  \notag \\
    &-p_{z\nu_1}^4 (A^2 - B^2)^2 + p_{z\nu_1}^2 \left[4A^2B^2p_{T\nu_1}^2 + 2C(A^2 + B^2)\right] -C^2 =0
\end{align}
$$

Equation (1) can be solved by $\texttt{Mathematica}$ or other tools, giving four solutions for $p_{z\nu_1}$. Please refer to bottom of [visualization](./visualize.ipynb) for the implementation of these solutions.