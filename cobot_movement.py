# Turn on the suction pump
def pump_on():
    # make position 2 work
    mc.set_basic_output(2, 0)
    # make position 5 work
    mc.set_basic_output(5, 0)


# stop the suction pump
def pump_off():
    # Stop position 2 from working
    mc.set_basic_output(2, 1)
    # Stop position 5 from working
    mc.set_basic_output(5, 1)


# function to make the cobot move

def move(list1):
    
    for i in range(len(list1)):
        mc.send_coords(list1[i],10,1)
        time.sleep(10)
        pump_on()
    pump_off()
    

#function to convert pixel coordinate system to cobots coordinate system

def convert(x,y):
    angle1 = -90
    angle2 = 180
    
    t1 = math.radians(angle1)
    t2 = math.radians(angle2)
   
    cost1 = math.cos(t1)
    sint1 = math.cos(t1)
    
    cost2 = math.cos(t2)
    sint2 = math.sin(t2)
    
    A = np.array([[1, 0 ,0 ][cost2, -sint2 ,0],[sint2, cost2 ,0]])
    B = np.array([[cost1, -sint1,0][sint2, cost2,0],[1,0,0]])
    
    C = np.dot(A, B)
    
    C = np.array([[0,-1,0],[1,0,0],[0,0,-1]])
    
    D = np.array([[0.26*x] ,[0.26*y],[1]])
    
    D = np.dot(C, D)
    
    return D[0]-112.8, D[1]+76
