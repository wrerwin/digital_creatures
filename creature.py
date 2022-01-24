class Creature:
    """
    """
    def __init__(self, x, y, brain):
        self.x = x
        self.y = y
        self.velocity_x = 0
        self.velocity_y = 0
        self.num_neighbors = None
        self.brain = brain
    
    def get_state(self):
        # x, y, and other variables *related to this creature only*
        # are always up to date
        # variables related to other creatures, such as number of 
        # neighbors, must be updated using the sense methods
        state = [self.x,
                 self.y,
                 self.velocity_x,
                 self.velocity_y]
        return state
        
    def move(self, dx, dy):
        """
        This moves the creature by updating its x and y positions
        The velocities of the creature are also updated (this assumes
        the move operation is called every time step)
        """
        self.x = self.x+dx
        self.y = self.y+dy
        self.velocity_x = dx
        self.velocity_y = dy
