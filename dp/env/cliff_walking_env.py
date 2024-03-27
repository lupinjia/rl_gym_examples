# Based on the code of Hands-on-RL(https://github.com/boyu-ai/Hands-on-RL/blob/main/%E7%AC%AC4%E7%AB%A0-%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%E7%AE%97%E6%B3%95.ipynb)

class CliffWalkingEnv:
    """ Self-implemented Cliff-Walking environment. """
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # columns of the grid world
        self.nrow = nrow  # rows of the grid world
        self.action_space = ['^', 'v', '<', '>']  # [up, down, left, right]
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]
        self.P = self.createP()

    def createP(self):
        # P is a 3D list. First dimension is the number of states (nrow * ncol),
        # second dimension is the number of actions (4), 
        # and third dimension is a list of tuples (p, next_state, reward, done).
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)] 
        # origin of the grid world is at the top left corner
        # 4 kind of actions, change[0]: up, change[1]: down, change[2]: left, change[3]:right
        # change[a][0] is the change in x-coordinate for action a, change[a][1] is the change in y-coordinate for action a.
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):  # i is the y-coordinate of the state
            for j in range(self.ncol):  # j is the x-coordinate of the state
                for a in range(4):  # a is action subscript
                    # The bottom row is the cliff and the goal state, except the leftmost column.
                    # When at the cliff or goal state, no action can be taken, and the reward is 0.
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0,
                                                    True)]
                        continue
                    # Other states
                    # min() and max() are used to bound the x and y coordinates within the grid world.
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))  # Choose the next state's x coordinate based on the action
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))  # Choose the next state's y coordinate based on the action
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    # If the next state on the cliff or the goal state, the episode is done.
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # If the next state is cliff, the reward is -100.
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P