# Based on the code of Hands-on-RL(https://github.com/boyu-ai/Hands-on-RL/blob/main/%E7%AC%AC4%E7%AB%A0-%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%E7%AE%97%E6%B3%95.ipynb)

class CliffWalkingEnv:
    """ 悬崖漫步环境"""
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # 定义网格世界的列
        self.nrow = nrow  # 定义网格世界的行
        self.action_space = ['^', 'v', '<', '>']  # 定义动作空间
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.P = self.createP()

    def createP(self):
        # 初始化
        #  P为3维列表,第一维长度为网格的行数*列数,表示不同的当前状态
        #  第二维长度为4,表示4种动作
        #  第三维为列表,包含四元组(p, next_state, reward, done)
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)] 
        # 4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4): # a为动作下标
                    # 位置在悬崖或者目标状态,因为无法继续交互,任何动作奖励都为0
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0,
                                                    True)]
                        continue
                    # 其他位置
                    # min()和max()函数用于限制坐标范围
                    next_x = min(self.ncol - 1, max(0, j + change[a][0])) # 根据动作的x坐标变化决定下一个状态的x坐标
                    next_y = min(self.nrow - 1, max(0, i + change[a][1])) # 根据动作的y坐标变化决定下一个状态的y坐标
                    next_state = next_y * self.ncol + next_x # 计算下一个状态的索引
                    reward = -1
                    done = False
                    # 下一个位置在悬崖或者终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # 下一个位置在悬崖
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P