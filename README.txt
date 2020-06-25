# minimization
a python code that uses the randomness of neural network training to find the ground state of a harmonic oscillator and its energy

you can run all the programs in colab (google colaboratory)

# MINIMIZATION PROGRAMS
1. minimization.py
latest version of the minimisation program
2. minimization_precision_multiplot.ipynb
this notebook plots the precision of the minimization with respect to the CPU time
3. minimization_other_potentials.ipynb
this notebook computes the ground states of V(x) = |x| and of V(x) = x^2 + a*exp(-b*x^2)
4. minimization_loss.ipynb
this notebook studies the network\'92s loss, iterations, convergence and CPU time

# OTHER PROGRAMS
5. loss_epoch.ipynb
this notebook studies a neural network trying to reproduce the ground state of a harmonic oscillator and plots the loss with respect to the epochs
6. scipy_tensorflow_interpolations.ipynb
this notebook compares the cubic interpolations of SciyPy and Tensorflow
7. energy_histogram.ipynb
this notebook makes a neural network reproduce a target, computes the energies (total, potential, kinetic) and the norm of the predictions of the network and then creates a histogram of those values
8. derivatives_of_neural_network_approximation.ipynb
this notebook uses a neural network to reproduce a gaussian curve and then computes derivatives (first and second) of the gaussian curve and of its reproduction