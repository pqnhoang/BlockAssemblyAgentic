CODE = '''
** Role **: You are an expert software programmer. Your task is to convert the given step-by-step plan into executable Python code using the Structure class.

** Important instructions **:
1. You will be given assembly instructions for building structures with blocks, cylinders, and other objects. Use the Structure class to implement these instructions.
2. Your primary responsibility is to translate assembly plans into Python code that creates the specified structures.
3. Follow the step-by-step plan precisely, implementing each action in the correct order.
4. Write code that correctly positions objects, checks stability, and handles any required simulations.

## Structure Class API:

```python
class Structure:
    """
    A class for physics-based simulation of various objects.
    
    Methods
    -------
    create_cylinder(target_: str, x: float, y: float, z: float)
        Creates a cylinder at the specified position
    create_block(target_: str, x: float, y: float, z: float)
        Creates a rectangular block at the specified position
    create_soft_sphere(target_: str, x: float, y: float, z: float)
        Creates a soft deformable sphere at the specified position
    create_muscle(target_: str, x: float, y: float, z: float)
        Creates a muscle object that can be actuated
    get_structure_image()
        Returns top and left view images of the current structure
    check_collision(target_: str, x: float, y: float, z: float)
        Checks for collisions at the specified position
    check_stability(target_: str, x: float, y: float, z: float)
        Checks stability of the structure
    simple_query(question: str)
        Returns the answer to a basic question about the structure
    llm_query(question: str)
        Returns a detailed answer about the structure using LLM
    """
    
    def create_cylinder(self, target_: str, x: float, y: float, z: float):
        """Creates a cylinder object at the specified position.
        
        Examples
        --------
        >>> # Create a cylinder for a table leg
        >>> def execute_command(feed_back_image):
        >>>     structure = Structure("table", available_blocks, "table", 1, feed_back_image)
        >>>     structure.create_cylinder("leg", 0.25, 0.1, 0.02)
        >>>     return structure
        """
        
    def create_block(self, target_: str, x: float, y: float, z: float):
        """Creates a rectangular block at the specified position.
        
        Examples
        --------
        >>> # Create a block for the table top
        >>> def execute_command(feed_back_image):
        >>>     structure = Structure("table", available_blocks, "table", 1, feed_back_image)
        >>>     structure.create_block("tabletop", 0.0, 0.0, 0.5)
        >>>     return structure
        """
        
    def create_soft_sphere(self, target_: str, x: float, y: float, z: float):
        """Creates a soft deformable sphere at the specified position.
        
        Examples
        --------
        >>> # Create a soft sphere and actuate it
        >>> def execute_command(feed_back_image):
        >>>     structure = Structure("simulation", available_blocks, "", 0, feed_back_image)
        >>>     sphere = structure.create_soft_sphere("soft_sphere", 0.5, 0.2, 0.3)
        >>>     structure.scene.build()
        >>>     for i in range(10):
        >>>         actu = np.array([0.2 * np.sin(0.1 * np.pi * i)])
        >>>         sphere.set_actuation(actu)
        >>>         structure.scene.step()
        >>>     return structure
        """
        
    def create_muscle(self, target_: str, x: float, y: float, z: float):
        """Creates a muscle object that can be actuated.
        
        Examples
        --------
        >>> # Create a muscle for simulation
        >>> def execute_command(feed_back_image):
        >>>     structure = Structure("muscle_sim", available_blocks, "", 0, feed_back_image)
        >>>     muscle = structure.create_muscle("muscle", 0.3, 0.3, 0.001)
        >>>     return muscle
        """
        
    def get_structure_image(self):
        """Returns top and left view images of the current structure.
        
        Examples
        --------
        >>> # Get images of the structure from two angles
        >>> def execute_command(feed_back_image):
        >>>     structure = Structure("visualization", available_blocks, "", 0, feed_back_image)
        >>>     structure.create_block("block", 0.0, 0.0, 0.0)
        >>>     structure.scene.build()
        >>>     structure.scene.step()
        >>>     top_img, left_img = structure.get_structure_image()
        >>>     return top_img, left_img
        """
        
    def check_stability(self, target_: str, x: float, y: float, z: float):
        """Checks if the structure would remain stable after adding a block.
        
        Examples
        --------
        >>> # Check if adding a block would maintain stability
        >>> def execute_command(feed_back_image):
        >>>     structure = Structure("stability_test", available_blocks, "", 0, feed_back_image)
        >>>     structure.create_block("base", 0.0, 0.0, 0.0)
        >>>     is_stable = structure.check_stability("next_block", 0.05, 0.0, 0.04)
        >>>     if is_stable:
        >>>         structure.create_block("next_block", 0.05, 0.0, 0.04)
        >>>     return is_stable
        """
        
    def simple_query(self, question: str):
        """Returns the answer to a basic question about the structure.
        
        Examples
        --------
        >>> # Ask about the structure
        >>> def execute_command(feed_back_image):
        >>>     structure = Structure("query_test", available_blocks, "", 0, feed_back_image)
        >>>     structure.create_block("block", 0.0, 0.0, 0.0)
        >>>     answer = structure.simple_query("How many blocks are in the structure?")
        >>>     return answer
        """
        
    def llm_query(self, question: str):
        """Returns a detailed answer about the structure using an LLM.
        
        Examples
        --------
        >>> # Ask a complex question about the structure
        >>> def execute_command(feed_back_image):
        >>>     structure = Structure("analysis", available_blocks, "", 0, feed_back_image)
        >>>     structure.create_block("block1", 0.0, 0.0, 0.0)
        >>>     structure.create_block("block2", 0.0, 0.0, 0.04)
        >>>     structure.scene.build()
        >>>     structure.scene.step()
        >>>     analysis = structure.llm_query("Does each block is stable or not?")
        >>>     return analysis
        """
  Write a function using Python and the Structure class (above) that could be executed to provide an answer to the query. 


### Examples
{example}

Plan at this step: {plan}
** Expected format output begin with **
def execute_command(image):
'''

EXAMPLES_CODER = '''
### Example 1
Plan:
Step 1: Initialize the structure for block simulation.
Step 2: Create a cylinder at position (0.25, 0.1, 0.02).
Step 3: Create a block at position (0.5, 0.1, 0.02).
Step 4: Create a soft sphere at position (0.5, 0.2, 0.3).
Step 5: Build the scene and run the simulation with the soft sphere being actuated.
Step 6: Return images of the final state.
A: ```
def execute_command(feed_back_image):
    available_blocks = os.path.join(BASE_PATH, 'data/simulated_blocks.json')
    structure = Structure("block", available_blocks, "", 0, feed_back_image)
    structure.create_cylinder("cylinder", 0.25, 0.1, 0.02)
    structure.create_block("block", 0.5, 0.1, 0.02)
    robot_mpm = structure.create_soft_sphere("soft_sphere", 0.5, 0.2, 0.3)
    structure.scene.build()
    for i in range(10):
        actu = np.array([0.2 * (0.5 + np.sin(0.01 * np.pi * i))])
        robot_mpm.set_actuation(actu)
        structure.scene.step()
    
    top_img, left_img = structure.get_structure_image()
    return top_img, left_img
    ```

### Example 2
Plan:
Step 1: Initialize a structure for building a tower.
Step 2: Create a base block at the origin.
Step 3: Add three more blocks stacked vertically on top of the base block.
Step 4: Build the scene and run the simulation for 10 steps.
Step 5: Check the stability of the tower.
Step 6: Return the top and left view images of the tower.
A: ```
def execute_command(feed_back_image):
    available_blocks = os.path.join(BASE_PATH, 'data/simulated_blocks.json')
    structure = Structure("tower", available_blocks, "tower", 4, feed_back_image)
    
    # Create base block
    structure.create_block("block1", 0.0, 0.0, 0.0)
    
    # Stack blocks vertically
    structure.create_block("block2", 0.0, 0.0, 0.04)
    structure.create_block("block3", 0.0, 0.0, 0.08)
    structure.create_block("block4", 0.0, 0.0, 0.12)
    
    # Build and run simulation
    structure.scene.build()
    for i in range(10):
        structure.scene.step()
    
    # Check stability
    stability = structure.llm_query("Is the tower stable?")
    
    # Get images
    top_img, left_img = structure.get_structure_image()
    return top_img, left_img, stability
    ```
'''