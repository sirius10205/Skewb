import numpy as np
import random
import cv2
import time
import pandas as pd

#define the array of skewb
#Arranged in such order: Center, Right-Up, Right-Down, Left-Down, Left_Up,
skewb = np.array([
    [ 1,  2,  3,  4,  5], #[1]Left 
    [ 6,  7,  8,  9, 10], #[2]Front
    [11, 12, 13, 14, 15], #[3]Down
    [16, 17, 18, 19, 20], #[4]up
    [21, 22, 23, 24, 25], #[5]Right
    [26, 27, 28, 29, 30]  #[6]Back
])
Skewb_original = skewb.copy()
#Take the counterclockwise Rotations of four verteices: FUR, FLU, BRU, FDR.
'''
R = (6, 16, 21)(2, 30, 12)(7, 18, 25)(8, 19, 22)(10, 17, 24),
L = (1, 6, 16)(2, 19, 10)(3, 20, 7)(5, 18, 9)(15, 27, 25),
B = (16, 26, 21)(5, 13, 7)(17, 30, 22)(18, 27, 23)(20, 29, 25),
D = (6, 21, 11)(3, 18, 29)(7, 23, 15)(8, 24, 12)(9, 25, 13),
'''
Gen= np.array([[2,4,5],[2,1,4],[6,5,4],[2,5,3]])
Gen = Gen -1
move_dict = ["R", "L", "B", "D"]
dir = ["clockwise", "counterclockwise"]

move_list_df = pd.read_csv('move_list.csv')
move_list = np.array(move_list_df.values.tolist())

def Permute_3(n, t):
    if t == 0:
        return np.array([n[2],n[0],n[1]])
    if t == 1:
        return np.array([n[1],n[2],n[0]])
    
def Rotation(Vertix, t, s):
    #vertix in {0,1,2,3}
    #t takes 0 or 1 which means the clockwise or counterclockwise rotation
    status = s.copy()
    center = np.array([ status[Gen[Vertix][0]][0],
                        status[Gen[Vertix][1]][0],
                        status[Gen[Vertix][2]][0] ])
    center = np.array(Permute_3(center, t))
    status[Gen[Vertix][0]][0] = center[0]
    status[Gen[Vertix][1]][0] = center[1]
    status[Gen[Vertix][2]][0] = center[2]
    if Vertix == 0:
        # FRU
        p1 = np.array([status[0][1],status[5][4],status[2][1]])
        p2 = np.array([status[1][1],status[3][2],status[4][4]])
        p3 = np.array([status[1][2],status[3][3],status[4][1]])
        p4 = np.array([status[1][4],status[3][1],status[4][3]])
        p1 = Permute_3(p1,t)
        p2 = Permute_3(p2,t)
        p3 = Permute_3(p3,t)
        p4 = Permute_3(p4,t)
        status[0][1] = p1[0]
        status[5][4] = p1[1]
        status[2][1] = p1[2]
        status[1][1] = p2[0]
        status[3][2] = p2[1]
        status[4][4] = p2[2]
        status[1][2] = p3[0]
        status[3][3] = p3[1]
        status[4][1] = p3[2]
        status[1][4] = p4[0]
        status[3][1] = p4[1]
        status[4][3] = p4[2]
    elif Vertix == 1:
        # FLU
        p1 = np.array([status[0][1],status[3][3],status[1][4]])
        p2 = np.array([status[0][2],status[3][4],status[1][1]])
        p3 = np.array([status[0][4],status[3][2],status[1][3]])
        p4 = np.array([status[2][4],status[5][1],status[4][4]])
        p1 = Permute_3(p1,t)
        p2 = Permute_3(p2,t)
        p3 = Permute_3(p3,t)
        p4 = Permute_3(p4,t)
        status[0][1] = p1[0]
        status[3][3] = p1[1]
        status[1][4] = p1[2]
        status[0][2] = p2[0]
        status[3][4] = p2[1]
        status[1][1] = p2[2]
        status[0][4] = p3[0]
        status[3][2] = p3[1]
        status[1][3] = p3[2]
        status[2][4] = p4[0]
        status[5][1] = p4[1]
        status[4][4] = p4[2]
    elif Vertix == 2:
        # BRU
        p1 = np.array([status[0][4],status[2][2],status[1][1]])
        p2 = np.array([status[3][1],status[5][4],status[4][1]])
        p3 = np.array([status[3][2],status[5][1],status[4][2]])
        p4 = np.array([status[3][4],status[5][3],status[4][4]])
        p1 = Permute_3(p1,t)
        p2 = Permute_3(p2,t)
        p3 = Permute_3(p3,t)
        p4 = Permute_3(p4,t)
        status[0][4] = p1[0]
        status[2][2] = p1[1]
        status[1][1] = p1[2]
        status[3][1] = p2[0]
        status[5][4] = p2[1]
        status[4][1] = p2[2]
        status[3][2] = p3[0]
        status[5][1] = p3[1]
        status[4][2] = p3[2]
        status[3][4] = p4[0]
        status[5][3] = p4[1]
        status[4][4] = p4[2]
    elif Vertix == 3:
        # FDR
        p1 = np.array([status[0][2],status[3][2],status[5][3]])
        p2 = np.array([status[1][1],status[4][2],status[2][4]])
        p3 = np.array([status[1][2],status[4][3],status[2][1]])
        p4 = np.array([status[1][3],status[4][4],status[2][2]])
        p1 = Permute_3(p1,t)
        p2 = Permute_3(p2,t)
        p3 = Permute_3(p3,t)
        p4 = Permute_3(p4,t)
        status[0][2] = p1[0]
        status[3][2] = p1[1]
        status[5][3] = p1[2]
        status[1][1] = p2[0]
        status[4][2] = p2[1]
        status[2][4] = p2[2]
        status[1][2] = p3[0]
        status[4][3] = p3[1]
        status[2][1] = p3[2]
        status[1][3] = p4[0]
        status[4][4] = p4[1]
        status[2][2] = p4[2]
    return status

def Random_move(n, status):
    for i in range(n):
        i = i-1
        v = random.randint(0,3)
        t = random.randint(0,1)
        status = Rotation(v,t,status)
        print(move_dict[v],dir[t])
    return status

def Draw_skewb(status):
    canvas = np.zeros((300, 400, 3), dtype="uint8")
    #Define points
    l1 = []
    for i in range(0,3):
        l1.append((100+50*i,300))
    l2 = []
    for i in range(0,2):
        l2.append((100+100*i,250))
    l3 = []
    for i in range(0,9):
        l3.append((50*i,200))
    l4 = []
    for i in range(0,5):
        l4.append((100*i,150))
    l5 = []
    for i in range(0,9):
        l5.append((50*i,100))
    l6 = []
    for i in range(0,2):
        l6.append((100+100*i,50))
    l7 = []
    for i in range(0,3):
        l7.append((100+50*i,0))
    #Draw Centers
    ptc1 = np.array([l3[1],l4[1],l5[1],l4[0]])
    ptc2 = np.array([l3[3],l4[1],l5[3],l4[2]])
    ptc3 = np.array([l1[1],l2[0],l3[3],l2[1]])
    ptc4 = np.array([l5[3],l6[0],l7[1],l6[1]])
    ptc5 = np.array([l3[5],l4[2],l5[5],l4[3]])
    ptc6 = np.array([l3[7],l4[3],l5[7],l4[4]])
    cv2.fillConvexPoly(canvas, ptc1, Color_switch(status[0][0]))
    cv2.fillConvexPoly(canvas, ptc2, Color_switch(status[1][0]))
    cv2.fillConvexPoly(canvas, ptc3, Color_switch(status[2][0]))
    cv2.fillConvexPoly(canvas, ptc4, Color_switch(status[3][0]))
    cv2.fillConvexPoly(canvas, ptc5, Color_switch(status[4][0]))
    cv2.fillConvexPoly(canvas, ptc6, Color_switch(status[5][0]))
    #Draw Triangles
    t2 = np.array([l4[1],l5[2],l5[1]])
    t3 = np.array([l3[1],l3[2],l4[1]])
    t4 = np.array([l3[0],l3[1],l4[0]])
    t5 = np.array([l5[0],l5[1],l4[0]])
    cv2.fillConvexPoly(canvas, t2, Color_switch(status[0][1]))
    cv2.fillConvexPoly(canvas, t3, Color_switch(status[0][2]))
    cv2.fillConvexPoly(canvas, t4, Color_switch(status[0][3]))
    cv2.fillConvexPoly(canvas, t5, Color_switch(status[0][4]))
    t7 = np.array([l5[3],l5[4],l4[2]])
    t8 = np.array([l3[3],l3[4],l4[2]])
    t9 = np.array([l3[2],l3[3],l4[1]])
    t10 = np.array([l5[2],l5[3],l4[1]])
    cv2.fillConvexPoly(canvas, t7, Color_switch(status[1][1]))
    cv2.fillConvexPoly(canvas, t8, Color_switch(status[1][2]))
    cv2.fillConvexPoly(canvas, t9, Color_switch(status[1][3]))
    cv2.fillConvexPoly(canvas, t10, Color_switch(status[1][4]))
    t12 = np.array([l3[3],l3[4],l2[1]])
    t13 = np.array([l1[1],l1[2],l2[1]])
    t14 = np.array([l1[0],l1[1],l2[0]])
    t15 = np.array([l3[2],l3[3],l2[0]])
    cv2.fillConvexPoly(canvas, t12, Color_switch(status[2][1]))
    cv2.fillConvexPoly(canvas, t13, Color_switch(status[2][2]))
    cv2.fillConvexPoly(canvas, t14, Color_switch(status[2][3]))
    cv2.fillConvexPoly(canvas, t15, Color_switch(status[2][4]))
    t17 = np.array([l7[1],l7[2],l6[1]])
    t18 = np.array([l5[3],l5[4],l6[1]])
    t19 = np.array([l5[2],l5[3],l6[0]])
    t20 = np.array([l7[0],l7[1],l6[0]])
    cv2.fillConvexPoly(canvas, t17, Color_switch(status[3][1]))
    cv2.fillConvexPoly(canvas, t18, Color_switch(status[3][2]))
    cv2.fillConvexPoly(canvas, t19, Color_switch(status[3][3]))
    cv2.fillConvexPoly(canvas, t20, Color_switch(status[3][4]))
    t22 = np.array([l5[5],l5[6],l4[3]])
    t23 = np.array([l3[5],l3[6],l4[3]])
    t24 = np.array([l3[4],l3[5],l4[2]])
    t25 = np.array([l5[4],l5[5],l4[2]])
    cv2.fillConvexPoly(canvas, t22, Color_switch(status[4][1]))
    cv2.fillConvexPoly(canvas, t23, Color_switch(status[4][2]))
    cv2.fillConvexPoly(canvas, t24, Color_switch(status[4][3]))
    cv2.fillConvexPoly(canvas, t25, Color_switch(status[4][4]))
    t27 = np.array([l5[7],l5[8],l4[4]])
    t28 = np.array([l3[7],l3[8],l4[4]])
    t29 = np.array([l3[6],l3[7],l4[3]])
    t30 = np.array([l5[6],l5[7],l4[3]])
    cv2.fillConvexPoly(canvas, t27, Color_switch(status[5][1]))
    cv2.fillConvexPoly(canvas, t28, Color_switch(status[5][2]))
    cv2.fillConvexPoly(canvas, t29, Color_switch(status[5][3]))
    cv2.fillConvexPoly(canvas, t30, Color_switch(status[5][4]))
    #Draw boarders
    cv2.line(canvas, l2[0],l1[1],(255,255,255),1)
    cv2.line(canvas, l2[1],l1[1],(255,255,255),1)
    cv2.line(canvas, l2[0],l5[5],(255,255,255),1)
    cv2.line(canvas, l2[1],l5[1],(255,255,255),1)
    cv2.line(canvas, l3[1],l6[1],(255,255,255),1)
    cv2.line(canvas, l3[5],l6[0],(255,255,255),1)
    cv2.line(canvas, l4[0],l5[1],(255,255,255),1)
    cv2.line(canvas, l4[0],l3[1],(255,255,255),1)
    cv2.line(canvas, l3[5],l5[7],(255,255,255),1)
    cv2.line(canvas, l3[7],l5[5],(255,255,255),1)
    cv2.line(canvas, l4[4],l5[7],(255,255,255),1)
    cv2.line(canvas, l4[4],l3[7],(255,255,255),1)
    cv2.line(canvas, l7[1],l6[0],(255,255,255),1)
    cv2.line(canvas, l7[1],l6[1],(255,255,255),1)
    cv2.line(canvas, l1[0],l7[0],(255,255,255),1)
    cv2.line(canvas, l1[2],l7[2],(255,255,255),1)
    cv2.line(canvas, l3[0],l3[8],(255,255,255),1)
    cv2.line(canvas, l5[0],l5[8],(255,255,255),1)
    cv2.line(canvas, l3[6],l5[6],(255,255,255),1)
    cv2.imshow("Skewb", canvas) #10
    cv2.waitKey(0) #11

def Color_switch(n):
    red = (0, 0, 205)
    green = (0, 205, 0)
    blue = (205, 0, 0)
    white = (205,205,205)
    yellow = (0,255,255)
    orange = (0,165,255)
    colors = [white,red,green,blue,yellow,orange]
    
    c = colors[(n-1)//5]
    return c

def move_decode(status, path):
    s = status.copy()
    l = len(str(path))
    x = str(path)
    for i in range(l):
        k = int(x[i])
        if k == 8:
            k=0
        s = Rotation(k//2, k%2, s)
    return s

def Search(status):
    n_local = [1, 8, 48, 288, 1728, 10248, 59304]
    p_global = [0, 8, 56, 344, 2072, 12320, 71624]
    #Step forward
    s = status.copy()
    s_o = skewb.copy()
    step_count0 = 0
    step_count1 = 0
    list1 = np.expand_dims(status.copy(), axis = 0)
    for step_count in range(12):
        if step_count == 0 :
            if (status == Skewb_original).all():
                return [9,9]
        else:
            if step_count%2==1:
                #Step from original status
                step_count0 = step_count0 + 1
                move0 = move_list[
                    p_global[step_count0-1]: p_global[step_count0], :]
                new_s = move_decode(s_o, move0[0][3])
                
                list0 = np.expand_dims(move_decode(s_o, move0[0][3]), axis=0)
                for j in range(1, n_local[step_count0]):
                    new_s = move_decode(s_o, move0[j][3])
                    new_s = np.expand_dims(new_s, axis=0)
                    list0 = np.append(list0, new_s, axis=0)
                    
            else:
                #Step from present status
                step_count1 = step_count1 + 1
                move1 = move_list[
                    p_global[step_count1-1]: p_global[step_count1], :]
                new_s = move_decode(s, move1[0][3])
                
                list1 = np.expand_dims(move_decode(s, move1[0][3]), axis=0)
                for j in range(1, n_local[step_count1]):
                    new_s = move_decode(s, move1[j][3])
                    new_s = np.expand_dims(new_s, axis=0)
                    list1 = np.append(list1, new_s, axis=0)
            for m_0 in range(list0.shape[0]):
                for m_1 in range(list1.shape[0]):
                    if (list0[m_0,:,:]==list1[m_1,:,:]).all():
                        if step_count == 1:
                            return [move0[m_0, 3], 9]
                        else:
                            return[move0[m_0, 3], move1[m_1, 3]]

def Solve(status):
    s = status.copy()
    m_0 = Search(s)[0]
    m_1 = Search(s)[1]
    str1 = str(m_0)
    str2 = str(m_1)
    l = []
    for j in range(len(str2)):
        temp = int(str2[j])
        if temp==8:
            temp = 0
        l.append([temp//2, temp%2])
    for i in range(len(str1)):
        temp = int(str1[len(str1)-i-1])
        if temp==8:
            temp = 0
        if temp%2 == 1:
            l.append([temp//2, 0])
        else: 
            l.append([temp//2, 1])
    return np.array(l)

if __name__ == '__main__':

    #Run random test
    print("Random status:")
    num = 15 #test with 15 random moves 
    r = Random_move(num, skewb)
    Draw_skewb(r)
    T1 = time.time()
    l = Solve(r)
    T2 = time.time()
    print("Solved with",l.shape[0], "move(s)" )
    print("Time used:", T2 - T1, "second(s)")
    for i in range(l.shape[0]):
        r = Rotation(l[i,0],l[i,1],r)
        print(move_dict[l[i,0]], dir[l[i,1]])
        Draw_skewb(r)
    print("Done")