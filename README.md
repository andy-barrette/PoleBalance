Created by Andrew Barrette

Pole-balancing problem solved by AL (advantage learning) + SARSA (sate-action-reward-state-action).
The value function, V, for AL is approximated by a neural network 5 layers deep and 6 neurons wide. The width accounts for 3 inputs + 3 actions.
Replays (stochastically sampled experiences) are used to avoid unwanted correlations during training, which might lead to unlearning.

PLOTS:

Plots consist of top-left plot (cart and pole) and top-right (V for each of the 3 actions: move left, do nothing, move right).

The next 3 plots below that are:
(1)
Target V (green)
Approximated V (red)
(2)
Cart position (yellow)
Rod angle (green)
(3)
Reward vs time

CONTROLS:

space		Display only every 50th frame
p			Pause
-/=			Decrease/increase view scale
,/.			Nudge rod left/right
r			Reset
d			Print NN