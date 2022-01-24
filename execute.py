#from organism import world,organism
import brain
import creature
import matplotlib.pyplot as plt
import time
import numpy as np

### global settings ###
N_INPUTS = 4  # these 3 settings are for the brains
N_HIDDENS = 3 
N_OUTPUTS = 3
TIMESTEPS = 200
N_CREATURES = 300
NX = 201 # width of world (keeping it odd so I can center it on origin)
NY = 201
xmax = (NX-1)/2
xmin = -xmax
ymax = (NY-1)/2
ymin = -ymax
#######################

list_of_organisms = []
for i in range(N_CREATURES):
    x = np.random.randint(xmin,xmax)
    y = np.random.randint(ymin,ymax)
    # brain weights are randomized for now
    b = brain.Brain(N_INPUTS, N_HIDDENS, N_OUTPUTS, 
        np.random.uniform(low=-1, high=1, size=(N_HIDDENS, N_INPUTS)),
        np.random.uniform(low=-1, high=1, size=(N_OUTPUTS, N_HIDDENS)))
    list_of_organisms.append(creature.Creature(x, y, b))

plt.ion()
for t in range(TIMESTEPS):
    x_coords = []
    y_coords = []
    for i in range(len(list_of_organisms)):
        c = list_of_organisms[i]
        # this is where the perception and decision making happens
        ####################################################
        dx = np.random.choice((-2,-1,0,1,2))
        dy = np.random.choice((-2,-1,0,1,2))

        current_knowledge = c.get_state()
        desires = c.brain.evaluate(current_knowledge)
        dx = np.round(2*desires[0] + desires[2])
        dy = np.round(2*desires[1] + desires[2])
        #####################################################
        
        # tar pit
        if c.x > 0 and c.y > 0:
            dx = 0
            dy = 0
        # walls
        if c.x+dx > xmax or c.x+dx < xmin:
            dx = 0
        if c.y+dy > ymax or c.y+dy < ymin:
            dy = 0
        # other creature collision check (slow)
        for c2 in list_of_organisms:
            if c.x+dx==c2.x and c.y==c2.y:
                dx=0
                dy=0
                break
        # move (currently move must be called every time step, even if 
        # dx=dy=0, for velocity tracking to work correctly. See creature.py 
        c.move(dx,dy) 

        x_coords.append(c.x)
        y_coords.append(c.y)
        
    # plot_obj = plt.scatter(x_coords,y_coords)
    plt.scatter(x_coords,y_coords)
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    plt.draw()
    plt.pause(0.00001)
    plt.clf()
