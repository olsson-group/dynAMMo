from dynamics_utils.potential import LangevinSampler, OneDimensionalDoubleWellPotential





if __name__ == '__main__':
    import matplotlib.pyplot as plt

    potential = OneDimensionalDoubleWellPotential(a=5.0, b=0.0)

    # Initial position of the particle
    x0 = 0.5  # Example initial position

    # Initialize the Langevin sampler for the one-dimensional case
    sampler = LangevinSampler(potential, x0=x0, dt=1, kT=1.0, mGamma=1000.0)

    # Run the simulation
    x = sampler.run(50000)
    plt.hist(x, bins=50)
    plt.show()