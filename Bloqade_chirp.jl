import Pkg; 
Pkg.add("Yao")

Pkg.add("YaoPlots")

using Bloqade
using PythonCall
using KrylovKit
using SparseArrays
using Yao, YaoPlots
plt = pyimport("matplotlib.pyplot");



# specify a 1D chain of 9 atoms with open boundary conditions
nsites = 9
atoms = generate_sites(ChainLattice(), nsites, scale = 5.72); # atom separation of 5.72 um

# specify the adiabatic pulse sequence for Rabi frequency by using the built-in waveform function
total_time = 3.0; Ω_max = 2π * 4;
# piecewise_linear function creates a linear waveform
Ω = piecewise_linear(clocks = [0.0, 0.1, 2.1, 2.2, total_time], values = [0.0, Ω_max, Ω_max, 0, 0]);

# specify detuning sequence:
U1 = -2π * 10; U2 = 2π * 10;
Δ = piecewise_linear(clocks = [0.0, 0.6, 2.1, total_time], values = [U1, U1, U2, U2]);

# generate time-dependent Hamiltonian:
h = rydberg_h(atoms; Δ, Ω);

# specify all atoms to be in the |0> state initially
reg = zero_state(nsites);
# and set up the emulation problem by choosing an ODE solver:
prob = SchrodingerProblem(reg, total_time, h);
integrator = init(prob, Vern8());

# TimeChoiceIterator is used to measure the Rydberg density on each site at various times:
densities = []
for _ in TimeChoiceIterator(integrator, 0.0:1e-3:total_time)
    push!(densities, rydberg_density(reg))
end
D = hcat(densities...);


# plot Rydberg density for each site as a function of time:
fig, ax = plt.subplots(figsize = (10, 4))
shw = ax.imshow(real(D), interpolation = "nearest", aspect = "auto", extent = [0, total_time, 0.5, nsites + 0.5])
ax.set_xlabel("time (μs)")
ax.set_ylabel("site")
ax.set_xticks(0:0.2:total_time)
ax.set_yticks(1:nsites)
bar = fig.colorbar(shw, label="Rydberg density")
#fig.savefig("state_prep.png", bb_inches="tight")
fig

bitstring_hist(reg; nlargest = 20)

# calculate the expectation value of pauli_x matrix, for each of the 9 sites.
sigma_x_i = []
for i in 1:nsites
    append!(sigma_x_i, abs(expect(put(nsites,i=>X), reg)))
end

# plot the expectation value of pauli_x matrix for each site
fig, ax = plt.subplots(figsize = (8, 4))
ax.grid()
ax.scatter(collect(1:nsites)[1:2:end], sigma_x_i[1:2:end], c="b", label="|r>", zorder=2)
ax.scatter(collect(1:nsites)[2:2:end], sigma_x_i[2:2:end], c="r", label="|g>", zorder=2)

ax.set_xlabel("Site ")
ax.set_ylabel("<\sigma_i^x >")
ax.legend(title="Target state", loc = "upper right", title_fontsize=15, fontsize=15)
#fig.savefig("expectation_value.png")
fig

deltaE = zeros(0) # array will store the energy splitting for values of pulse detuning
Ω = 2π * 4
Δ_step = 30
Δ = LinRange(-2π * 10, 2π * 10, Δ_step);

for ii in 1:Δ_step
    h_ii = rydberg_h(atoms; Δ = Δ[ii], Ω) # create the Rydberg Hamiltonian
    h_m = mat(h_ii) # convert the Hamiltonian into a matrix
    vals, vecs, info = KrylovKit.eigsolve(h_m, 2, :SR) # find the ground state eigenvalue and eigenvector
    append!(deltaE, vals[2]-vals[1])
end

min_dE = minimum(deltaE)/(2π)
println("Minimum energy-splitting = ", min_dE, "/2π MHz")
println("Minimum time = ", 1/(min_dE)^2)

# plot deltaE as a function of the pulse detuning
fig, ax = plt.subplots(figsize = (10, 4))
plt.axhline(y=minimum(deltaE)/(2π), color='r', linestyle="dashed")
ax.plot(Δ / 2π, deltaE / 2π)
ax.set_xlabel("Δ/2π (MHz) ")
ax.set_ylabel("ΔE/2π (MHz)")
ax.grid()
#fig.savefig("deltaE_vs_detuning.png", bb_inches="tight")
fig

"""
calc_deltaE calculates the minimum energy splitting between the ground and first-excited state
params:
    nsites: number of atoms in 1D chain (int)
returns:
    minimum energy splitting (real)
"""
function calc_deltaE(nsites)
    atoms = generate_sites(ChainLattice(), nsites, scale = 5.72);
    deltaE = zeros(0)
    Ω = 2π * 4
    Δ_step = 30
    Δ = LinRange(-2π * 10, 2π * 10, Δ_step);

    for ii in 1:Δ_step
        h_ii = rydberg_h(atoms; Δ = Δ[ii], Ω) # create the Rydberg Hamiltonian
        h_m = mat(h_ii) # convert the Hamiltonian into a matrix
        vals, vecs, info = KrylovKit.eigsolve(h_m, 2, :SR) # find the ground state eigenvalue and eigenvector
        append!(deltaE, (vals[2]-vals[1])/(2π))
    end
    return minimum(deltaE)
end

# calculate minimum deltaE for 1D chains with 1,3,...,15 atoms
deltaE = []
for i in collect(1:15)[1:2:end]
    append!(deltaE, calc_deltaE(i))
end

xs = LinRange(0.1, 15, 100)

fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (12, 4))
ax1.scatter(collect(1:15)[1:2:end], deltaE.^2, zorder=2, label="1D Chain", c="black")
ax1.grid()
ax1.set_xlabel("Number of atoms (n)")
ax1.set_ylabel("ΔE/2π (MHz)")
ax1.legend(loc = "upper right", fontsize=15)
ax1.set_ylim(0,20)

ax2.set_yscale("log")
ax2.scatter(collect(1:15)[1:2:end], (deltaE.^2).^(-1), zorder=2, label="1D Chain", c="black")
ax2.plot(xs, xs.^2, zorder=2, label="O( n^2 )", c="navy")
ax2.plot(xs, 2 .^xs, zorder=2, label="O( 2^n )", c="firebrick")
ax2.grid()
ax2.set_xlabel("Number of atoms (n)")
ax2.set_ylabel("t (us)")
ax2.legend(loc = "upper left", fontsize=15)

#fig.savefig("deltaE_and_time.png", bb_inches="tight")
fig


