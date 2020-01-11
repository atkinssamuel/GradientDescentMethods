### Project Description:
This project includes a Python implementation of an MLP used to contrast gradient descent with stochastic gradient descent. Both gradient descent and stochastic gradient descent were implemented for the pumadyn32nm dataset. 
### pumadyn32nm Dataset:
#### Gradient Descent:
The first 1000 training data points of the pumadyn32nm dataset were used. The following learning rates were tested: [0.1, 0.01, 0.001, 0.0001, 0.00001]. The error plots pertaining to each learning rate are illustrated in the Figures below:

From the Figures above, small learning rates did not allow the network to learn quick enough. Larger learning rates, however, caused rapid convergence. A considerably large learning rate of 0.1 did not cause oscillation or instability in the error plot evidenced by Figure _. A reason for this could be that the input and output data have a trivial correlation and a smooth functional space. 

#### Stochastic Gradient Descent:
For stochastic gradient descent, the following learning rates were used: [0.01, 0.001, 0.0001, 0.00001]. Just as above, the error plots are illustrated below:

Oscillation and instability are present in every plot. From the results above, we can conclude the stability that comes from the nature of gradient descent was neccessary for the application of an MLP to the pumadyn32nm dataset.

### iris Dataset:
#### Gradient Descent:

#### Stochastic Gradient Descent:



How to run:
- The gradientDescent file contains the entire implementation of the assignment
	- The function "gradientDescent" pertains to question 1 only
	- The function logisticGradientDescent" pertains to question 2 only
- Currently, both functions are uncommented and ready to run in the __main__ block
	- To run just one question, simply call that function
- All neccessary figures will autogenerate and the relevant values will print
- The files for question 1 will automatically save in folders titled "results/GD" and "results/SGD"
- The files for question 2 will automatically save in folders titled "results/LogGD" and "results/LogSGD"
	- These results are already contained in the submitted zip folder