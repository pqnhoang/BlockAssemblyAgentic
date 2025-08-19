from src.toolset import IsometricImage
import numpy as np
import json

structure_names = [
    "Giraffe", "Path", "Spaceship", "Earthquake", "Fountain", "Roman Column", 
    "Stairs", "UFO", "Bench", "Cross", "Island", "Circus", "Cow", "Billboard", 
    "Rook Chess Piece", "Train Station", "Queen Chess Piece", "Key", "Universe", 
    "Train", "Cliff", "Cinema", "Aquarium", "Bird", "Asteroid", "Ant", "Pig", 
    "Playground", "Pool", "Ramp", "Tower", "Rainbow", "Alley", "Medal Podium", 
    "Motorcycle", "Palace", "Square", "Hotel", "Bishop Chess Piece", "Star", 
    "Box", "Burger", "Satellite", "Lightening", "Turtle", "Door", "Statue", 
    "Fence", "Heart", "Ferris Wheel", "Sun", "Waterfall", "Gate", "Column", 
    "Airport", "Dog", "Plane", "Monument", "Submarine", "Barrier", "Rain", 
    "Library", "Lighthouse", "Window", "Igloo", "Truck", "Mountain", "Windmill", 
    "Spider", "Knight Chess Piece", "Pedestal", "Factory", "Cabin", "Archway", 
    "Cloud", "Forest", "Duck", "Hut", "Chair", "Ladder", "Road", "Theater", 
    "Tent", "Underpass", "Beach", "Street", "Swing", "Banner", "Bee", "Pyramid", 
    "Stool", "Water Tower", "Lollipop", "Tractor", "Table", "Horse", "TV", 
    "Roller Coaster", "Tunnel", "Car", "Sheep", "Swan", "House", "Whale", 
    "Astronaut", "Volcano", "Jungle", "Sandbox", "Flag", "Traffic Light", 
    "Top Hat", "Boat", "Temple", "Dinosaur", "Elephant", "Pillar", "Eruption", 
    "Lamp", "Museum", "Alien", "Arrow", "Wall", "Street Lamp", "Barricade", 
    "Space Station", "Cave", "Clock", "Gazebo", "Sign", "Crosswalk", "Stadium", 
    "Greek Column", "Park", "Bridge", "Castle", "Slide", "Flower", "Helicopter", 
    "Robot", "Tree", "Ladybug", "Rocket", "Pawn Chess Piece", "Obelisk", 
    "Arch", "Highway", "Overpass", "Shark", "Butterfly", "Orchard", 
    "King Chess Piece", "Bicycle", "Kite", "Cat", "Store", 
    "Empire State Building", "Parthenon", "Eifel Tower", "Taj Mahal", 
    "Ceiling Fan", "Soccer Goal", "Tote Bag", "Well", "Teddy Bear", 
    "Computer Monitor", "Coaster", "Closed Box", "Flower Pot", "Shelf", 
    "Filament Roll", "Screw", "Couch", "Sofa", "Calipers", "Glasses", 
    "Letter A", "Letter B", "Letter C", "Letter D", "Letter E", "Letter F", 
    "Letter G", "Letter H", "Letter I", "Letter J", "Letter K", "Letter L", 
    "Letter M", "Letter N", "Letter O", "Letter P", "Letter Q", "Letter R", 
    "Letter S", "Letter T", "Letter U", "Letter V", "Letter W", "Letter X", 
    "Letter Y", "Letter Z"
]

def get_rating(response):
    """Extract rating number from response"""
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1:
            data = json.loads(response[start:end])
            return data.get('rating', 0)
        return 0
    except:
        return 0

if __name__ == "__main__":
    ratings = []
    
    for i, name in enumerate(structure_names):
        try:
            print(f"[{i+1}/200] {name}")
            
            # Generate structure
            img = IsometricImage(object_name=name)
            img.describe_object()
            img.make_plan(img.main_llm_context[-1]['content'])
            img.order_blocks(img.main_llm_context[-1]['content'])
            img.decide_position(img.main_llm_context[-1]['content'])
            img.make_structure(img.positions)
            img.refine_structure(img.blocks)
            
            # Get rating
            rating = get_rating(img.get_structure_rating())
            ratings.append(rating)
            print(f"  Rating: {rating}")
            
        except Exception as e:
            print(f"  Error: {e}")
            ratings.append(0)
    
    # Calculate average
    valid = [r for r in ratings if r > 0]
    if valid:
        print(f"\nAverage Rating: {np.mean(valid):.2f}")
        print(f"Valid ratings: {len(valid)}/200")