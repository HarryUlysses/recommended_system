import redis


class MyRedis:

    def __init__(self, host='127.0.0.1', port='6379', db=0):

        self.redis = redis.Redis(host, port, db)
    def setRmse(self,key,args):
        self.redis.set(key,args)

    def getRmse(self,key):
        value = self.redis.get(key)
        if value[0] == '(':
            rmse = float(value[1:-4])
        else:
            rmse = float(value)
        return rmse

    def setList(self, key, *args):
        i = 0
        for arg in args:
            i += 1
            self.redis.set(key + str(i), arg)

    def getList(self, key):
        list3 = []
        list4 = []

        list0 = str(self.redis.get(key + str(1)))[1:-1].split(", ")
        list1 = str(self.redis.get(key + str(2)))[1:-1].split(", ")
        for str0 in list0:
            list3.append(float(str0))
        for str0 in list1:
            list4.append(float(str0))
        return list3, list4
