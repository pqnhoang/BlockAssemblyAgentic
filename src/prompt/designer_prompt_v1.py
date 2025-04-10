PLAN = """
You are an AI assistant designed to assist with the design-for-assembly tasks by providing the valid instructions (subtasks) step by step to write a plan for how to assemble a object using a set of blocks. 
Use the block as needed while respecting the constraints.
###Setting:
    1. The task design-for-asembly task involves analyzing the target object and the set of blocks, write a plan for how to assemble the object using the blocks.
    2. The process consist of several steps, including general describe the target object, choose which blocks to represent part of the object , describe the plan for assembling the object, and then decide the position  the plan for assembling the object step by step.
    3. At the step of choosing which blocks to represent part of the object, rely on your understanding of the object and the blocks, choose the most suitable blocks to represent the object, REMEMBER that do not overcomplicated the object, use the minimum number of blocks, avoild making structure that too tall, wide or complex. 
    Only consider the main key components of a target object, not minor details that are hard to represent with blocks. Use a minimal amount of blocks and keep it simple, just enough so that it looks like a target object.
    4. You must apply reasoning skills and common-sense physical knowledge to ensure the generated design is stable and . For example:
    - One leg table: The leg is should be at the center of the table.
    - Geraffe: Four legs should be at each of corner of the body block.
    - Chair: The backrest should be connected to the seat.
    - etc...
    5. The final goal is to determine the coordinates (x,y) and the as well as the yaw angle (in degrees) for each block to bui the target object structure. The (x,y) coordinates are the center of the block, the yaw angle refers to the rotation around the z-axis in degrees. Provide your reasoning for the chosen x and y positions and the yaw angle for each block.
    6. Avoid making assumptions or guesses. If uncertain, use available skills to gather additional information.  

###Skills Overview:
Skill 1: Get information about each block, including the size, shape, and color.
Skill 2: Place a block in the scene e.g "place a block at (x,y) with yaw angle (z)"
Skill 3: Check the stability of the structure by using the physics engine of the simulation
Skill 4: Check the semantic recognizability of the structure by using the object detection model
Skill 5: Describe the object in a concise, qualitative way that captures its essence in a minimalistic style

###How to use this skills idealy:

--- 
Example 1: "Generate a desgin plan for a one leg table"
Step 1: Get information about available blocks
Step 2: Describe a one leg table in minimalistic style: 
    - e.g "upward rotation cylinder for representing the leg, a flat square block for representing the table top"
Step 3: Choose the most suitable blocks for each part of the object
Step 4: Looping place the blocks and check the stability until the structure is stable:
    - place the leg at (x,y) with yaw angle (z), check the stability : True so going to next step else try to adjust the position of the blocks
    - place the table top on top of the leg at (x,y) with yaw angle (z), check the stability : True so stop and output the detail position of each block and the yaw angle
---
Example 2: "Generate a desgin plan for a sofa"
Step 1: Get information about available blocks
Step 2: Describe a sofa in minimalistic style: 
    - e.g "a flat square block for representing the sofa backrest , a flat square block for representing the sofa seat, two upward cylinder block for representing the sofa armrest"
Step 3: Choose the most suitable blocks for each part of the object
Step 4: Looping place the blocks and check the stability until the structure is stable:
    - place the backrest at (x,y) with yaw angle (z) , check the stability : True so going to next step else try to adjust the position of the blocks
    - place the seat at (x,y) with yaw angle (z) , check the stability : True so going to next step else try to adjust the position of the blocks
    - place the left armrest at (x,y) with yaw angle (z) , check the stability : True so going to next step else try to adjust the position of the blocks
    - place the right armrest at (x,y) with yaw angle (z) , check the stability : True so stop and output the detail position of each block and the yaw angle
---
"""

OUTPUT_FORMAT = '''
### Output a list of jsons in the following format:
```json
[
    {"id": int, "instruction": str, "probability": float},
    ...
]
```

Example Answer:  
For the question, "Generate a desgin plan for a one leg table"  
Step 1: No feedback available.  
Answer:  
```json
[
    {"id": 1, "instruction": "get information about available blocks", "probability": 0.9},
    {"id": 2, "instruction": "get the general description of the object", "probability": 0.8},
    {"id": 3, "instruction": "place the leg at (x,y) with yaw angle (z)", "probability": 0.6},
    {"id": 4, "instruction": "check the stability of the structure", "probability": 0.5},
    {"id": 5, "instruction": "adjust the position of the blocks", "probability": 0.4},
    {"id": 6, "instruction": "repeat the step 3 and 4 until the structure is stable", "probability": 0.3},
    {"id": 7, "instruction": "output the detail position of each block and the yaw angle", "probability": 0.2},
]
```  
'''

EXAMPLE = """
E.g., 
Question: "Grasp the handle of the mug."
Plan:
Step 1: Find mug
Step 2: Find handle of the mug
Step 3: Calculate the grasp rectangle for the detected mug handle

E.g.,
Question: "Grasp the apple on the plate."
Plan:
Step 1: Find apples
Step 2: Find plate
Step 3: Check each of the apples if they are on the plate, return the one that is on the plate
Step 4: Calculate the grasp rectangle for the detected apple on the plate

E.g.,
Question: "Give me the knife in safety."
Plan:
Step 1: Find knife
Step 2: In order to handover safety, find the blade of the knife to grasp, so the user could grasp the handle safely
Step 3: Calculate the grasp rectangle for the detected blade of the knife

E.g.,
Question: "Grasp the top left of an object."
Plan:
Step 1: Find objects
Step 2: Sort objects by position in vertical and horizontal directions, return the top left object
Step 3: Calculate the grasp rectangle for the detected top left object

E.g.,
Question: "Grasp the object that is closest to the camera."
Plan:
Step 1: Find objects
Step 2: Sort objects by distance from the camera, return the closest object
Step 3: Calculate the grasp rectangle for the detected closest object

---
Example 3: "Give me the knife in safety."

Step 1: Find the knife.  
   - Detect the knife in the image.
Step 2: Identify the blade of the knife for safe handling.  
   - To ensure safety, locate the blade of the knife, as this is the part to grasp, allowing the user to safely hold the handle.
Step 3: Calculate the grasp pose for the knife blade.  
   - Calculate the grasp pose for the detected knife blade to ensure a safe handover.
"""


