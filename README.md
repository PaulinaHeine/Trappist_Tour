#######
Multi objective optimization challenge
#######

##########Problem:##########

TRAPPIST-1 system needs to be explored. This should be done wihth the least amount of delta V and time.
The trajectory is determined by a 34 dimensional Vector.
	- 27 astrophysical parameters (floats)
	- 7  integers that determies the sequence of visiting the planets

The goal is to find the set of solutions that forms the best possible pareto front.
#################Solution approach:######################

Decomposing the vector into two partial problems:
 1.: Find the optimal sequential order of visiting the exoplanets, determined by the seven categorical integer variables.
 2.: Determine the optimal values for the continuous astronomical parameters in order to minimize both objectives.

Solving p1:
	Selecting X random permutations and optimize the 27 other values with pymoo.
	Scoring the sequences.
	Evaluating which sequences are better and whcih worse.
	Depending on that evaluation, analyse which sequences of two score high.
	Form new permutations out of the good sequences of two.
	
Solving p2:
	Optimize the formed permutations with pymoo. Using different algotithms. Than searching for the best pareto front.
	
########Result########

This approach results in a score of -3 440 253.65

	
	
	
