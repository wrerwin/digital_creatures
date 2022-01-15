from organism import world,organism
import matplotlib.pyplot as plt
import time
import numpy as np

new_world = world(n_organisms = 100)       
list_of_organisms = new_world.organism_states
x_coords = []
y_coords = []

for organism_obj in list_of_organisms:
    x_coords.append(organism_obj.x)
    y_coords.append(organism_obj.y)

timesteps = 100


plt.ion()

for t in range(timesteps):
    x_coords = []
    y_coords = []
    for i,_ in enumerate(list_of_organisms):
        # TODO: Instead of random choices for dx and dy, this is where the perception and decision making
        # will happen.
        dx = np.random.choice((-2,-1,0,1,2))
        dy = np.random.choice((-2,-1,0,1,2))

        current_knowledge = list_of_organisms[i].perceive()
        # list_of_organisms[i].make_decision()
        # dummy code below to illustrate what this may look like 
        ####################################################
        if current_knowledge['x_pos'] > 50 and current_knowledge['y_pos'] > 50:
            dx = 0
            dy = 0
        list_of_organisms[i].move(dx,dy)
        ####################################################
        x_coords.append(list_of_organisms[i].x)
        y_coords.append(list_of_organisms[i].y)
        
    # plot_obj = plt.scatter(x_coords,y_coords)
    plt.scatter(x_coords,y_coords)
    plt.xlim([0,100])
    plt.ylim([0,100])
    plt.draw()
    plt.pause(0.00001)
    plt.clf()
