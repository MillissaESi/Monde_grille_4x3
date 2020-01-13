# Monde_grille_4x3

![Grille](/images/fig1.png)

The purpose of this task is to implement dynamic programming algorithms based on policy or iteration value, in order to resolve grille world problem that consists of maximizing a score from a start cell to a final cell in the grille. 
To resolve this problem, we need first to define the PDM representing this environment: 
* S: the set of the problem states. In this case, each cell in the grille will represent a state. All the cells that represents a wall, will not be considered in this set.
* A : there are four actions: “up”, “down”, “right”, “left”
* T: the transition matrix. In this problem a transition from state s1 to s2 by executing action “a” corresponds to:
0,8 if we reach s2 by running action “a”
0.1 if s2 is in a vertical angle compared to the move “a”
    * R: rewards, vary between the cells. 
    * the discount factor 
    
Dynamic programming algorithms are used to calculate the optimal valuation to find the optimal policy to apply. For both algorithms implemented for this task, we will calculate the valuation values until we guarantee an epsilon-convergence of the valuation function. 

We will compare between “Policy iteration” and “iteration value” algorithms based on three different reward cases:

#### 1. reward negative (-0,04): 

![Policy + value iteration results](/images/fig2.png)

Policy + value iteration results

#### 2. reward is a great negatif number: 

Policy iteration: 

![Policy iteration](/images/fig3.png)

#### 3. reward positive (great numbers), Reward = 100 :
  
Policy iteration:

![Policy iteration](/images/fig4.png)
                  
Value iteration: 

![Value iteration](/images/fig5.png)

                  


