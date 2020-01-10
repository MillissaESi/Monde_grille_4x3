import numpy as np

class grille:
    def __init__(self, lignes, columns, start, terminals, walls, reward, gam):
        self.states = {}
        for i in range(lignes):
            for j in range(columns):
                self.states[(i,j)] = reward
        for key, value in terminals.items():
            self.states[key] = value
        for wall in walls:
            self.states.pop(wall)
        self.start = start
        self.gam = gam
        self.actions = ['up', 'down', 'right', 'left']

    def transitionPolicy(self,a, state, v):
        t = 0
        if a == "up":
            t += 0.8 * v[(state[0] - 1, state[1])] if (state[0] - 1, state[1]) in self.states else  0.8 * v[state]
            t += 0.1 * v[(state[0], state[1] - 1)] if (state[0] , state[1] - 1) in self.states else  0.1 * v[state]
            t += 0.1 * v[(state[0], state[1] - 1)] if (state[0] , state[1] + 1) in self.states else  0.1 * v[state]
        elif a == "down":
            t += 0.8 * v[(state[0] + 1, state[1])] if (state[0] + 1, state[1]) in self.states else 0.8 * v[state]
            t += 0.1 * v[(state[0], state[1] - 1)] if (state[0], state[1] - 1) in self.states else 0.1 * v[state]
            t += 0.1 * v[(state[0], state[1] - 1)] if (state[0], state[1] + 1) in self.states else 0.1 * v[state]
        elif a == "right":
            t += 0.8 * v[(state[0], state[1] + 1)] if (state[0], state[1] +1) in self.states else 0.8 * v[state]
            t += 0.1 * v[(state[0] - 1, state[1])] if (state[0] - 1, state[1]) in self.states else 0.1 * v[state]
            t += 0.1 * v[(state[0] + 1, state[1])] if (state[0] + 1, state[1]) in self.states else 0.1 * v[state]
        else :
            t += 0.8 * v[(state[0], state[1] - 1)] if (state[0], state[1] - 1) in self.states else 0.8 * v[state]
            t += 0.1 * v[(state[0] - 1, state[1])] if (state[0] - 1, state[1]) in self.states else 0.1 * v[state]
            t += 0.1 * v[(state[0] + 1, state[1])] if (state[0] + 1, state[1]) in self.states else 0.1 * v[state]
        return t

    def valueIteration(self, eps):
        # initialization
        v = {}
        for state in self.states:
            v[state] = 0
        # evaluation
        start = True
        while start or delta < eps:
            start = False
            delta = 0
            for state in self.states:
                value = v[state]
                v[state] = max(self.states[(state[0] - 1,state[1])] + self.gam * v[(state[0] - 1,state[1])] if (state[0] - 1,state[1]) in self.states else 0, self.states[(state[0] + 1,state[1])] + self.gam * v[(state[0] + 1,state[1])] if (state[0] + 1,state[1]) in self.states else 0, self.states[(state[0] ,state[1] - 1)] + self.gam * v[(state[0] ,state[1] - 1)] if (state[0] ,state[1] - 1) in self.states else 0, self.states[(state[0] ,state[1] + 1)] + self.gam * v[(state[0] ,state[1] + 1)] if (state[0],state[1] + 1) in self.states else 0 )
                delta = max(delta, abs(value - v[state]))
        return v

    def policyIteration(self , eps):
        # pick an arbitrary policy
        pi={}
        for state in self.states:
            a = self.actions[np.random.randint(4)]
            pi[(a, state)] = 1
        v_k = {state: 0 for state in self.states}
        v_newk = {state: 0 for state in self.states}
        while True:
            # policy evaluation
            for state in self.states:
                for a in self.actions:
                    v_newk[state] += pi[(a,state)] * (self.states[state] + self.gam * self.transitionPolicy(a, state, v_k))
            # policy improvement
            for state in self.states:
                v = 0
                for a in self.actions:
                    pi[(a,state)] = 0
                    k = self.states[state] + self.gam * self.transitionPolicy(a, state, v_newk)
                    if k > v:
                        v = k
                        action = a
                pi[(action,state)] = 1
            # check convergence
            err = 0
            for state in self.states:
                err = max( abs(v_newk[state] - v_k[state]), err)
            if err < eps:
                return v_newk

"""
  Environment 
"""
gam = 0.6
lignes = 4
columns = 3
start = (0,0)
terminals = {(2,1): 1, (3,4): -1}
reward = -0.04
walls = [(2,2)]
world = grille(lignes, columns, start, terminals, walls, reward,  gam)
"""
  policy iteration
"""
eps = 0.1
v_opt = world.policyIteration(eps)
print("optimal value ", v_opt)


