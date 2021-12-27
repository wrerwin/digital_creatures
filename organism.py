from numpy.random import uniform
import settings

class organism():
    def __init__(self,name = None):
        self.x = uniform(settings.x_min,settings.x_max)
        self.y = uniform(settings.y_min,settings.y_max)
        self.name = name

        self.brain = None

    def update_state(self):
        '''
        update the state of the organism based on external stimuli
        '''

    def reproduce(self):
        '''
        generate a new organism if applicable, enables passing of old genes and mutations
        '''
        return child_organism
        
class world():
    '''
    Is this the best way to handle a collection of organisms? 
    How are they stored? Must they stay in memory? 
    '''
    def __init__(self,n_organisms):
        self.n_organisms = n_organisms
        self.organism_states = self.build_initial_config()

    def build_initial_config(self):
        '''
        Build the inital configuration of the world, including all organism states. Store all of the organisms somehow. 
        '''
        list_of_organisms = []
        for _ in range(self.n_organisms):
            list_of_organisms.append(organism())
        return list_of_organisms

    def update_organisms(self):
        '''
        updates the state of all organisms in the world based on organism state and world state
        '''