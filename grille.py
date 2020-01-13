import numpy as np

class grille:
    def __init__(self, lignes, columns, start, terminals, walls, reward, gam):
        self.states = {}
        for i in range(lignes):
            for j in range(columns):
                self.states[(i,j)] = reward
        for key, value in terminals.items():
            self.states[key] = value
        self.terminals = set(terminals.keys())
        for wall in walls:
            self.states.pop(wall)
        self.start = start
        self.gam = gam
        self.actions = ['up', 'down', 'right', 'left']

    def Transitions(self,s1, s2 , a):
            prob = 0
            if (s2 != s1):
                if (s2 == (s1[0] - 1 , s1[1]) and  a == "up")  or  ( s2 == (s1[0] + 1, s1[1]) and a == "down") or ( s2 == (s1[0], s1[1] - 1) and a == "left") or ( s2 == (s1[0], s1[1] + 1) and a == "right"):
                    prob += 0.8
                if ((s2 == (s1[0], s1[1] - 1) or s2 == (s1[0], s1[1] + 1)) and  (a == "up" or a == "down" )) or  ((a == "right" or a == "left") and (s2 == (s1[0] - 1, s1[1]) or s2 == (s1[0] + 1, s1[1]))):
                    prob += 0.1
            elif (s2 == s1) :
                if ( a == "up" and (s1[0] - 1 , s1[1]) not in self.states) or ( a == "down" and (s1[0] + 1 , s1[1]) not in self.states) or ( a == "right" and  (s1[0], s1[1] + 1) not in self.states) or ( a == "left" and  (s1[0], s1[1] - 1) not in self.states):
                    prob += 0.8
                if a == "up":
                    if ((s1[0], s1[1] - 1) not in self.states and (s1[0], s1[1] + 1) not in self.states) :
                        prob += 0.2
                    elif (s1[0], s1[1] - 1) not in self.states or (s1[0], s1[1] + 1) not in self.states :
                        prob += 0.1
                if a == "right" :
                    if ((s1[0] - 1, s1[1] ) not in self.states and (s1[0] +1 , s1[1] ) not in self.states):
                        prob += 0.2
                    elif ((s1[0] - 1, s1[1] ) not in self.states or (s1[0] +1 , s1[1] ) not in self.states):
                        prob += 0.1
                if a == "down" :
                    if ((s1[0], s1[1] - 1) not in self.states and (s1[0], s1[1] + 1) not in self.states):
                        prob += 0.2
                    elif (s1[0], s1[1] - 1) not in self.states or (s1[0], s1[1] + 1) not in self.states:
                        prob += 0.1
                if a == "left" :
                    if ((s1[0] - 1, s1[1] ) not in self.states and (s1[0] +1 , s1[1] ) not in self.states):
                        prob += 0.2
                    elif ((s1[0] - 1, s1[1] ) not in self.states or (s1[0] +1 , s1[1] ) not in self.states):
                        prob += 0.1
            return prob

    def transitionPolicy(self,a, state, v):
        output = 0
        for st in self.states:
            output +=  self.Transitions(state, st, a) * v[st]
        return output

    def valueIteration(self, eps):
        # initialization
        v = {}
        for state in self.states:
            v[state] = self.states[state] if state in self.terminals else 0
        # evaluation
        while True:
            delta = 0
            for state in self.states:
                if state in self.terminals:
                    continue
                value = v[state]
                sums = [v[state]]
                for a in self.actions:
                    va = 0
                    for st in self.states:
                        va +=  self.Transitions(state,st, a ) * (self.states[state] + self.gam * v[st] )
                    sums.append(va)
                v[state] = max(sums)
                delta = max(delta, abs(value - v[state]))
            if delta < eps:
                break
        # calculate the final policy
        pi = {}
        for state in self.states:
            val = - 1000
            for a in self.actions:
                pi[(a, state)] = 0
                k = self.states[state] + self.gam * self.transitionPolicy(a, state, v)
                if k > val:
                    val = k
                    action = a
            pi[(action, state)] = 1
        return pi, v

    def policyIteration(self , eps):
        # pick an arbitrary policy
        pi = {}
        for state in self.states:
            for a in self.actions:
                pi[(a,state)] = 0
            a = self.actions[np.random.randint(4)]
            pi[(a, state)] = 1
        # initilize value iteration
        v = {state: 0 for state in self.states}
        for state in self.terminals:
            v[state] = self.states[state]
        while True:
            delta = 0
            # policy evaluation
            for state in self.states:
                if state in self.terminals:
                    continue
                new = 0
                for a in self.actions:
                    new += pi[(a,state)] * (self.states[state] + self.gam * self.transitionPolicy(a, state, v))
                delta = np.maximum(np.abs(new - v[state]), delta)
                v[state] = new
            if delta < eps:
                break
            # policy improvement
            for state in self.states:
                old_k = [ "" , - 1000]
                for a in self.actions:
                    pi[(a,state)] = 0
                    k = self.states[state] + self.gam * self.transitionPolicy(a, state, v)
                    old_k = [a,k] if k > old_k[1] else old_k
                pi[(old_k[0], state)] = 1
        return pi, v

"""
  Environment 
"""
gam = 0.9
lignes = 4
columns = 3
start = (0,0)
terminals = {(3,1): -1, (3,2): 1}
reward = -100
walls = [(1,1)]
world = grille(lignes, columns, start, terminals, walls, reward,  gam)

"""
  policy iteration
"""


eps = 0.001
pi, v_opt = world.policyIteration(eps)
print("optimal value ", v_opt)
print(pi)




eps = 0.001
pi, v_opt = world.valueIteration(eps)
print("optimal value ", v_opt)
print(pi)




"""
T = []
a = "left"
for state in world.states:
    print(state)
    summ = []
    for st in world.states:
        summ.append(world.Transitions(state, st, a))
    T.append(summ)
for t in T:
    print(t)
"""





