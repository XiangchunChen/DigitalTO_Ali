class EdgeDevice:
    '所有子任务的基类'

    def __init__(self, deviceId, cpuNum, waitTime, resource):
        self.deviceId = deviceId
        self.cpuNum = cpuNum
        self.waitTime = waitTime
        self.resource = resource
        self.taskNum = 0 #一个Device上最多运行2个任务

    def setWaitTime(self, waitTime):
        self.waitTime = waitTime

    def setResource(self, resource):
        self.resource = resource
        
    def setTaskNum(self, num):
        self.taskNum = num
        
    def getTaskNum(self):
        return self.taskNum

    def printInfo(self):
        print("deviceId:"+str(self.deviceId)+",cpuNum:"+
                              str(self.cpuNum)+",waitTime:"+str(self.waitTime))
