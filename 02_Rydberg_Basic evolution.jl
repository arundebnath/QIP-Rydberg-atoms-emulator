println("Hello")

VERSION

using Bloqade

nsites = 07;

atoms = generate_sites(ChainLattice(), nsites, scale = 5.72)


# hamiltonian specifying the parameters

h = rydberg_h(atoms;  Δ=1.2*2π, Ω=4*2π, ϕ=2.1)

# initial state creation
reg = zero_state(nsites)

# time evolution time duration

prob = SchrodingerProblem(reg, 1.2, h)

# integrator choice

integrator = init(prob, Vern8());
emulate!(prob);



# checking population

rydberg_populations = map(1:nsites) do i
    rydberg_density(prob.reg, i)
end


