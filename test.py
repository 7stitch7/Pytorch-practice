A = [(0,2,0),(3,7,1),(2,3,2),(7,9,2),(0,3,3),(8,9,4)]



A = [(0,0,0),(0,0,1),(0,0,2),(0,0,3)]

def getKey(item):
    return item[1]
def Laser_Cannon(Alian):
    l = sorted(Alian, key=getKey)
    print(l)
    A = []
    C = []
    X = l[0][0]
    A = [l[0]]
    for i in range(1,len(l)):
        if l[i][0] <= A[-1][1]:
            if l[i][0] >= A[-1][0] and l[i][0] > l[i-1][0]:
                X = l[i][0]
                C.append(X)
        if l[i][0] > A[-1][1]:
            A.append(l[i])
    if X < A[-1][0]:
        C.append(A[-1][0])

    if len(C) == 0:
        C.append(X)

    return len(C),C,A

print(Laser_Cannon(A))
