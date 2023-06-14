# Quantum algorithms for neutral atom-based information processors

Here, I provide a set of elementary codes that provide a basic introduction to the Rydberg atom-based quantum computational platform. 

$$
\mathcal{H}(t) = \sum_j \frac{\Omega_j(t)}{2} \left( e^{i \phi_j(t) } | g_j \rangle  \langle r_j | + 
e^{-i \phi_j(t) } | r_j \rangle  \langle g_j | \right) - \sum_j \Delta_j(t) n_j + \sum_{j < k} V_{jk} n_j n_k
$$

 - The governing Hamiltonian encodes the problem to the Hardware. \\
 - The first term provides explicit external control. The last two terms, characterized by parametric variables, encode a graph-based problem into the hardware.
 - The time dependence of the parametric variables allows the solution space to be reached.
