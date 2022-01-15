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
        dx = np.random.choice((-1,0,1))
        dy = np.random.choice((-1,0,1))
        list_of_organisms[i].move(dx,dy)
        x_coords.append(list_of_organisms[i].x)
        y_coords.append(list_of_organisms[i].y)
    
    # plot_obj = plt.scatter(x_coords,y_coords)
    plt.scatter(x_coords,y_coords)
    plt.xlim([0,100])
    plt.ylim([0,100])
    plt.draw()
    plt.pause(0.0001)
    plt.clf()
