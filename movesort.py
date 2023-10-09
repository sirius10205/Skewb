import numpy as np
import time
import pandas as pd
import SkewbCount as sc

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

#Define move list with repeated elements.
status_list0 = np.expand_dims(sc.skewb.copy(),axis=0)     #1
status_list1 = []               #8
status_list2 = []               #48
status_list3 = []               #288
status_list4 = []               #1728
status_list5 = []               #10248
status_list6 = []               #59304
move_list1 = []
move_list2 = []
move_list3 = []
move_list4 = []
move_list5 = []
move_list6 = []

#Step 1
list = []
for m in range(4):
    for t in range(2):
        s = sc.Rotation(m,t,skewb)
        list.append(s)
        k = 2*m+t
        if k == 0:
            k = 8
        move_list1.append([m,t,2*m+t,k])
status_list1 = np.array(list)
status_list = np.expand_dims(status_list0[0,:,:], axis=0)
status_list = np.append(status_list, status_list1, axis =0)
#Step 2
list = []
for n in range(status_list1.shape[0]):
    for m in range(4):
        for t in range(2):
            if (m != move_list1[n][0]):
                s = sc.Rotation(m,t,status_list1[n,:,:])
                list.append(s)
                k = 2*m+t
                if k == 0:
                    k = 8
                move_list2.append([m,t,n,10*move_list1[n][3]+k])
status_list2 = np.array(list)
status_list = np.append(status_list, status_list2, axis =0)
#Step 3
list = []
for n in range(status_list2.shape[0]):
    for m in range(4):
        for t in range(2):
            if (m != move_list2[n][0]):
                s = sc.Rotation(m,t,status_list2[n,:,:])
                list.append(s)
                k = 2*m+t
                if k == 0:
                    k = 8
                move_list3.append([m,t,n,10*move_list2[n][3]+k])
status_list3 = np.array(list)
status_list = np.append(status_list, status_list3, axis =0)
#Step 4
list = []
for n in range(status_list3.shape[0]):
    for m in range(4):
        for t in range(2):
            if m != move_list3[n][0]:
                s = sc.Rotation(m,t,status_list3[n,:,:])
                list.append(s)
                k = 2*m+t
                if k == 0:
                    k = 8
                move_list4.append([m,t,n,10*move_list3[n][3]+k])
status_list4 = np.array(list)
status_list = np.append(status_list, status_list4, axis =0)
#From here there will be more repeated elements.

#Step 5
#list_5 contains 120 repeated elements in itself, but not in the other lists.
list = []
for n in range(status_list4.shape[0]):
    for m in range(4):
        for t in range(2):
            flag = False
            if m != move_list4[n][0]:
                s = sc.Rotation(m,t,status_list4[n,:,:])
                list.append(s)
                k = 2*m+t
                if k == 0:
                    k = 8
                move_list5.append([m,t,n,10*move_list4[n][3]+k])
status_list5 = np.array(list)
p5 = np.unique(status_list5, return_index=True, axis=0)[1]
# np.unique returns array without repeated elements but sorted, 
# now restore the right order
move_list5_n = [move_list5[p5[0]]]
new_5 = np.expand_dims(status_list5[p5[0],:,:],axis=0)
for i in range(1,p5.shape[0]):
    s_n = np.expand_dims(status_list5[p5[i],:,:],axis=0)
    new_5 = np.append(new_5, s_n, axis = 0)
    move_list5_n.append(move_list5[p5[i]])
status_list5 = np.array(new_5)
move_list5 = move_list5_n
status_list = np.append(status_list, status_list5, axis =0)

#Step 6
list = []
for n in range(status_list5.shape[0]):
    for m in range(4):
        for t in range(2):
            flag = False
            if m != move_list5[n][0]:
                s = sc.Rotation(m,t,status_list5[n,:,:])
                list.append(s)
                k = 2*m+t
                if k == 0:
                    k = 8
                move_list6.append([m,t,n,10*move_list5[n][3]+k])
status_list6 = np.array(list)
p6 = np.unique(status_list6, return_index=True, axis=0)[1]
move_list6_n = [move_list6[p5[0]]]
new_6 = np.expand_dims(status_list6[p6[0],:,:],axis=0)
for i in range(1,p6.shape[0]):
    s_n = np.expand_dims(status_list6[p6[i],:,:],axis=0)
    new_6 = np.append(new_6, s_n, axis = 0)
    move_list6_n.append(move_list6[p6[i]])
status_list6 = np.array(new_6)
move_list6 = np.array(move_list6_n)
status_list = np.append(status_list, status_list6, axis =0)

#search repeated elements
p6_n = []
for j in range(status_list6.shape[0]):
    s_flag = False
    for i in range(status_list.shape[0]):
        if (status_list[i,:,:] == status_list6[j,:,:]).all():
            s_flag = True
            break
    if s_flag:
        p6_n.append(j)
pd_data = pd.DataFrame(p6_n)
pd_data.to_csv("index6.csv", header=None, index=False)

#p6_n stores the indeces of repeated elements, which occured in step 4 and 5.
df = pd.read_csv('index6.csv')
p6_n = df.values.tolist()

#Draw_skewb(status_list6[37808,:,:])
#print(move_list6[37808,:])

status_list6 = np.delete(status_list6, p6_n, axis = 0)
move_list6 = np.delete(move_list6, p6_n, axis = 0)

#The global status and move records of all 6 steps without repeated elements.
move_list = np.array(move_list1)
move_list = np.append(move_list, move_list2, axis=0)
move_list = np.append(move_list, move_list3, axis=0)
move_list = np.append(move_list, move_list4, axis=0)
move_list = np.append(move_list, move_list5, axis=0)

move_list = np.append(move_list, move_list6, axis=0)
status_list = np.append(status_list, status_list6, axis =0)
#Write lists to .csv file
pd.DataFrame(move_list).to_csv("move_list.csv", index=None)