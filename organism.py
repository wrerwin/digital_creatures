from numpy.random import uniform
import settings

class organism():
    def __init__(self,name = None):
        self.x = uniform(settings.x_min,settings.x_max)
        self.y = uniform(settings.y_min,settings.y_max)
        self.name = name
        
        self.brain = initialize_brain()
        self.knowledge = None
        self.next_actions = None
        
    def update_state():
        '''
        update the state of the organism based on external stimuli
        '''
        self.act()
        self.knowledge = perceive()
        self.next_actions = action()
        
        
    def perceive(perception_type):
        '''
        defines the ability for an organism to percieve aspects of the state of its environment
        TODO: At this stage the type of information we want is limited to whether +/- x or y is advantageous (maybe this should happen in the brain?)
        '''
    
        return perceived_information

    def action():

       '''
       defines the ability for an organism to make outwardly perceptible actions
       '''
        # pseudocode below
        possible_actions = ['x move','y move']
        action_steps_dict = {}
        for action in possible_actions:
            action_steps_dict[action] = perceive
        return action_steps_dict

    def act():
        '''
        implements the action that is defined in self.action() and updates the organisms coordinates (or other params affected by action)
        '''
        # pseudocode below
        organism.x = organism.x + organism.next_actions['x move']
        organism.y = organism.y + organism.next_actions['y move']
        organism.next_actions = None
 
        
    def initialize_brain():
        '''
        intialize a brain with appropriate connectivity

        TODO: 
        How will this work? Do we just build a small neural net with some activation function?
        How does this connect to actions. Where are possible actions/perceptions defined? (probably a utils file somewhere?)
        
        '''
        
        return brain
    
    def reproduce(reproduction_condition):
        '''
        evaluates a reproduction condition and generates a new organism if applicable
        enables passing of old genes and mutations
        '''
        child_organism = organism()
        child_organism.brain = build_child_brain()
        
        def build_child_brain():
            '''
            uses the parent(s) brain as the initial condition, introduces the possibility for mutation to connect to a new tool/perception.
            '''
            return child_brain
        
        return child_organism
        
class world():
    '''
    Is this the best way to handle a collection of organisms? 
    How are they stored? 
    '''

    def __init__(self,n_organisms):
        self.n_organisms = n_organisms
        self.organism_states = self.build_initial_config()

    def build_initial_config():
        '''
        Build the inital configuration of the world, including all organism states. Store all of the organisms somehow. 
        '''
        list_of_organisms = []
        for _ in range(self.n_organisms):
            list_of_organisms.append(organism())
        return list_of_organisms

    def update_organisms():
        '''
        updates the state of all organisms in the world based on organism state and world state. 
        calls organism.update_state() for all organisms
        
        TODO: handle this with a better data structure than a list suggested in the pseudocode 
        '''
        updated_organism_states = self.organism_states # placeholder
        # pseudocode below
#         updated_organism_states = []
#         for organism in self.organism_states:
#             organism.update_state()
#         updated_organism_states.append(organism)
        return updated_organism_states

    def reproduce_organisms(reproduction_condition):
        '''
        population level reproduction. Calls organism.reproduce(reporduction_condition) for all organisms
        '''
    