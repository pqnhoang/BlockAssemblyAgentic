PLAN = """
You are an AI assistant designed to assist with the design-for-assembly tasks by providing the valid instructions (subtasks) step by step to write a plan for how to assemble the target object using a set of blocks. 
Use the block as needed while respecting the constraints.

###User Query:
{user_query}

###Previous Plan:
{planning}

###Observation:
{observation}

###Available Blocks:
{available_blocks}

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
Step 1: Get blocks information.
Step 2: Choose the appropriate block for each part of the object.
Step 3: Place each block follow the order of the plan. Each time you place a block, check the stability of the structure and feedback to the user.
Step 4: Output the result
---
Example 2: "Generate a desgin plan for a sofa"
Step 1: Get blocks information.
Step 2: Choose the appropriate block for each part of the object.
Step 3: Place each block follow the order of the plan. Each time you place a block, check the stability of the structure and feedback to the user.
Step 4: Output the result
---
###Required Output Format:
{return_format}

###Example Answer:
{examples}
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
    {"id": 2, "instruction": "get the general description of the one leg table", "probability": 0.8},
    {"id": 3, "instruction": "choose flat cylinder block to represent the table top", "probability": 0.7},
    {"id": 4, "instruction": "choose thin cuboid to represent the table leg", "probability": 0.7},
    {"id": 5, "instruction": "place the leg at (0, 0) with yaw angle 0", "probability": 0.6},
    {"id": 6, "instruction": "check the stability of the leg placement", "probability": 0.5},
    {"id": 7, "instruction": "if stable, place the table top at (0, 0.5) with yaw angle 0", "probability": 0.4},
    {"id": 8, "instruction": "check the stability of the table structure", "probability": 0.3},
    {"id": 9, "instruction": "output the detail position of each block and the yaw angle", "probability": 0.2}
]
```
'''

EXAMPLE = """
"""


