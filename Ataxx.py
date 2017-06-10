#coding=utf-8
import numpy as np
import random

class Ataxx:
    # 移动方向 3*3 + 5*5
    delta = np.array([[1,1],[0,1],[-1,1],[-1,0],
                      [-1,-1],[0,-1],[1,-1],[1,0],
                      [2,0],[2,1],[2,2],[1,2],
                      [0,2],[-1,2],[-2,-2],[-2,-1],
                      [-2,0],[-2,-1],[-2,-2],[-1,-2],
                      [0,-2],[1,-2],[2,-2],[2,-1]])

    def __init__(self):
        self.gridInfo = np.zeros((7,7),dtype=np.int)
        self.gridInfo[0, 0] = 1
        self.gridInfo[6, 6] = 1
        self.gridInfo[6, 0] = -1
        self.gridInfo[0, 6] = -1
        self.currBotColor = 1 # 1为黑子 -1为白子 黑子先手
        self.blackCount=2
        self.whiteCount=2

    #判断是否在地图内
    def inMap(self, x, y):
        if x<0 or x>6 or y<0 or y>6:
            return False
        return True

    # 向Direction方向改动坐标，并返回是否越界
    def MoveStep(self,x,y,direction):
        x = x+self.delta[direction,0]
        y = y+self.delta[direction,1]
        return self.inMap(x,y)

    def ProcStep(self,x0,y0,x1,y1,color):
        if color==0:
            return False
        if x1==-1: # 无路可走,跳过此回合
            return True
        if not self.inMap(x0,y0) and not self.inMap(x1,y1): #越界
            return False
        if self.gridInfo[x0,y0]!= color:
            return False
        dx=0;dy=0;x=0;y=0;currCount=0;
        effectivePoints = np.zeros((8,2))
        dx = np.abs(x0-x1);dy=np.abs(y0-y1)
        if (dx==0 and dy==0) or dx>2 or dy>2: #保证区域为5*5范围内
            return False
        if self.gridInfo[x1,y1]!= 0: # 移动到的位置为空
            return False
        if dx == 2 or dy == 2: # 5*5范围时,移动棋子
            self.gridInfo[x0,y0]=0
        else:
            if color == 1:
                self.blackCount += 1
            else:
                self.whiteCount += 1

        self.gridInfo[x1,y1] = color
        for i in range(8):
            x = x1 + self.delta[i,0]
            y = x1 + self.delta[i,1]
            if not self.inMap(x,y):
                continue
            if self.gridInfo[x,y] == -color:
                effectivePoints[currCount,0]=x
                effectivePoints[currCount,1]=y
                currCount += 1
                self.gridInfo[x,y]=color
        if currCount !=0:
            if color ==1:
                self.blackCount += currCount
                self.whiteCount -= currCount
            else:
                self.blackCount -= currCount
                self.whiteCount += currCount
        return True

    def findLegalPoint(self,color):
        beginPos=np.zeros((1000,2),dtype=np.int)
        possiblePos = np.zeros((1000,2),dtype=np.int)
        posCount = 0
        for y0 in range(0,7):
            for x0 in range(0,7):
                if self.gridInfo[x0,y0] != color:
                    continue
                for k in range(24):
                    x1 = x0 + self.delta[k,0]
                    y1 = y0 + self.delta[k,1]
                    if not self.inMap(x1,y1):
                        continue
                    if self.gridInfo[x1,y1]!= 0:
                        continue
                    beginPos[posCount,0]=x0
                    beginPos[posCount,1]=y0
                    possiblePos[posCount,0]=x1
                    possiblePos[posCount,1]=y1
                    posCount += 1
        return beginPos,possiblePos,posCount

    def isTerminal(self):
        if self.whiteCount==0 or self.blackCount==0:
            return True
        return False

if __name__ == '__main__':
    env = Ataxx()
    for i in range(10):
        beginPos,possiblePos,posCount = env.findLegalPoint(1)
        count = random.randint(0,posCount)
        b_x0=beginPos[count,0]
        b_y0=beginPos[count,1]
        b_x1=possiblePos[count,0]
        b_y1=possiblePos[count,1]
        env.ProcStep(b_x0,b_y0,b_x1,b_y1,1)

        beginPos, possiblePos, posCount = env.findLegalPoint(-1)
        count = random.randint(0,posCount)
        w_x0 = beginPos[count, 0]
        w_y0 = beginPos[count, 1]
        w_x1 = possiblePos[count, 0]
        w_y1 = possiblePos[count, 1]
        env.ProcStep(w_x0, w_y0, w_x1, w_y1, -1)
        print env.whiteCount,env.blackCount
    print env.gridInfo







