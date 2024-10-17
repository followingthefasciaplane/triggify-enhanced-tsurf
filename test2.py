import sys
import re
import math
import logging
from time import sleep
from collections import defaultdict, deque

# Converts strings to numbers (int or float)
def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

class Vertex:
    def __init__(self, x, y, z):
        self.x = num(x)
        self.y = num(y)
        self.z = num(z)

    def __sub__(self, other):
        return Vertex(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other):
        return Vertex(self.x + other.x, self.y + other.y, self.z + other.z)

    def cross(self, other):
        return Vertex(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def normalize(self):
        mag = self.magnitude()
        if mag == 0:
            return self
        return Vertex(self.x / mag, self.y / mag, self.z / mag)

    def scale(self, scalar):
        return Vertex(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def magnitude(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def __repr__(self):
        return f"{self.x} {self.y} {self.z}"

    def __eq__(self, other):
        return isinstance(other, Vertex) and self.x == other.x and self.y == other.y and self.z == other.z

    def __hash__(self):
        return hash((self.x, self.y, self.z))

class Side:
    def __init__(self):
        self.id = None
        self.plane = []
        self.plane_str = ''
        self.vertices_plus = []  # Stores Vertex instances
        self.material = None
        self.uaxis = None
        self.vaxis = None
        self.rotation = None
        self.lightmapscale = None
        self.smoothing_groups = None

    def parse(self, data):
        self.id = data.get('id')
        plane_str = data.get('plane', '')
        self.plane_str = plane_str  # Store the original plane string
        # Parse plane
        plane_match = re.findall(
            r'\((-?\d+\.?\d*(?:e[+-]?\d+)?) (-?\d+\.?\d*(?:e[+-]?\d+)?) (-?\d+\.?\d*(?:e[+-]?\d+)?)\)', plane_str)
        if len(plane_match) != 3:
            raise ValueError(f"Invalid plane definition in side {self.id}: {plane_str}")
        self.plane = [Vertex(*coords) for coords in plane_match]

        # Parse vertices_plus if available
        vertices_plus_data = data.get('vertices_plus', {})
        if vertices_plus_data:
            vertices_list = vertices_plus_data.get('v', [])
            if not isinstance(vertices_list, list):
                vertices_list = [vertices_list]
            for vertex_str in vertices_list:
                coords = vertex_str.strip().split()
                if len(coords) != 3:
                    raise ValueError(f"Invalid vertex in vertices_plus of side {self.id}: {vertex_str}")
                self.vertices_plus.append(Vertex(*coords))

        self.material = data.get('material')
        self.uaxis = data.get('uaxis')
        self.vaxis = data.get('vaxis')
        self.rotation = data.get('rotation')
        self.lightmapscale = data.get('lightmapscale')
        self.smoothing_groups = data.get('smoothing_groups')

    def compute_normal(self):
        if len(self.plane) != 3:
            raise ValueError(f"Cannot compute normal for side {self.id}: plane does not have 3 vertices.")
        # Compute the normal vector of the plane
        p1, p2, p3 = [Vertex(v.x, v.y, v.z) for v in self.plane]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = v1.cross(v2).normalize()
        return normal

    def order_vertices(self, vertices, normal):
        # Compute centroid
        centroid = Vertex(
            sum(v.x for v in vertices) / len(vertices),
            sum(v.y for v in vertices) / len(vertices),
            sum(v.z for v in vertices) / len(vertices)
        )
        # Choose a reference vector
        ref_dir = (vertices[0] - centroid).normalize()
        # Compute angles from centroid
        def angle_from_centroid(v):
            vec = (v - centroid).normalize()
            # Calculate the angle using atan2 with respect to the reference direction
            cross = normal.cross(ref_dir).dot(vec)
            dot = ref_dir.dot(vec)
            angle = math.atan2(cross, dot)
            return angle
        # Sort vertices
        sorted_vertices = sorted(vertices, key=angle_from_centroid)
        return sorted_vertices

    def get_full_vertices(self):
        # Use vertices from vertices_plus if available
        if self.vertices_plus:
            if len(self.vertices_plus) != 4:
                logging.warning(f"Side {self.id} has {len(self.vertices_plus)} vertices in vertices_plus; expected 4. Skipping this side.")
                return []
            # Ensure vertices are ordered
            normal = self.compute_normal()
            ordered_vertices = self.order_vertices(self.vertices_plus, normal)
            return ordered_vertices
        else:
            # Fallback: Extract all unique vertices from the plane definitions
            plane_str = self.plane_str
            plane_match = re.findall(
                r'\((-?\d+\.?\d*(?:e[+-]?\d+)?) (-?\d+\.?\d*(?:e[+-]?\d+)?) (-?\d+\.?\d*(?:e[+-]?\d+)?)\)', plane_str)
            vertices = [Vertex(*coords) for coords in plane_match]
            if len(vertices) != 4:
                logging.warning(f"Side {self.id} has {len(vertices)} vertices in plane; expected 4. Skipping this side.")
                return []
            return vertices

class Solid:
    def __init__(self):
        self.id = None
        self.sides = []
        self.editor = {}

    def parse(self, data):
        self.id = data.get('id')
        sides = data.get('side')
        if not isinstance(sides, list):
            sides = [sides]
        for side_data in sides:
            side = Side()
            side.parse(side_data)
            self.sides.append(side)
        self.editor = data.get('editor', {})

class VMFParser:
    def __init__(self):
        self.entities = []
        self.world = {}
        self.versioninfo = {}
        self.viewsettings = {}
        self.cameras = {}
        self.cordons = {}
        self.solids = []
        self.world_solids_ids = []
        self.func_detail_solids = []

    def parse(self, file_path):
        with open(file_path, 'r') as f:
            content = f.read()

        tokens = self.tokenize(content)
        token_iter = iter(tokens)
        self.data = self.parse_block(token_iter)
        # Now extract entities and solids
        self.extract_data(self.data)

    def tokenize(self, content):
        # Improved tokenizer to handle nested blocks and strings
        token_pattern = r'"[^"]*"|[^\s{}"]+|{|}'
        tokens = re.findall(token_pattern, content)
        tokens = [token.strip() for token in tokens if token.strip()]
        return tokens

    def parse_block(self, tokens):
        data = {}
        key = None
        while True:
            try:
                token = next(tokens)
            except StopIteration:
                break

            if token == '}':
                return data
            elif token == '{':
                if key is None:
                    raise ValueError("Unexpected '{' without key or class name")
                else:
                    value = self.parse_block(tokens)
                    if key in data:
                        # Ensure data[key] is a list
                        if not isinstance(data[key], list):
                            data[key] = [data[key]]
                        data[key].append(value)
                    else:
                        data[key] = value
                    key = None
            elif token.startswith('"'):
                if key is None:
                    key = token.strip('"')
                else:
                    value = token.strip('"')
                    if key in data:
                        # Ensure data[key] is a list
                        if not isinstance(data[key], list):
                            data[key] = [data[key]]
                        data[key].append(value)
                    else:
                        data[key] = value
                    key = None
            else:
                if key is None:
                    key = token
                else:
                    # Handle key-value pairs without quotes
                    value = token
                    if key in data:
                        if isinstance(data[key], list):
                            data[key].append(value)
                        else:
                            data[key] = [data[key], value]
                    else:
                        data[key] = value
                    key = None
        return data

    def extract_data(self, data):
        self.versioninfo = data.get('versioninfo', {})
        self.viewsettings = data.get('viewsettings', {})
        self.world = data.get('world', {})
        self.cameras = data.get('cameras', {})
        self.cordons = data.get('cordons', {})

        # Parse solids in world
        world_solids = self.world.get('solid', [])
        if not isinstance(world_solids, list):
            world_solids = [world_solids]
        for solid_data in world_solids:
            solid = Solid()
            solid.parse(solid_data)
            self.solids.append(solid)
            self.world_solids_ids.append(solid.id)

        # Parse entities
        entities_data = data.get('entity', [])
        if not isinstance(entities_data, list):
            entities_data = [entities_data]
        for entity_data in entities_data:
            self.entities.append(entity_data)
            classname = entity_data.get('classname', '')
            # Check if the entity is a func_detail
            if classname == 'func_detail':
                # Parse solids in func_detail
                func_detail_solids = entity_data.get('solid', [])
                if not isinstance(func_detail_solids, list):
                    func_detail_solids = [func_detail_solids]
                for solid_data in func_detail_solids:
                    solid = Solid()
                    solid.parse(solid_data)
                    self.solids.append(solid)
                    self.func_detail_solids.append(solid)
            else:
                # Parse solids in other entities if needed
                pass

def generate_trigger_multiple(solid_id_counter, face_id_counter, side, height=4, targetname=None, wait=None):
    """
    Generates a trigger_multiple entity for a single side.
    Ensures that the solid has exactly six sides with four vertices each.
    """
    # Get all unique vertices of the base face
    base_vertices = side.get_full_vertices()
    if len(base_vertices) != 4:
        logging.warning(f"Skipping side {side.id} because it does not have exactly 4 vertices.")
        return None, face_id_counter

    # Compute the normal vector of the face using the ordered vertices
    try:
        normal = side.compute_normal()
    except ValueError as e:
        logging.error(e)
        return None, face_id_counter

    # Create the top vertices by moving along the normal vector
    top_vertices = [v + normal.scale(height) for v in base_vertices]

    # Calculate centroid of the base face for the origin
    centroid = Vertex(
        sum(v.x for v in base_vertices) / len(base_vertices),
        sum(v.y for v in base_vertices) / len(base_vertices),
        sum(v.z for v in base_vertices) / len(base_vertices)
    )
    origin = f"{centroid.x} {centroid.y} {centroid.z}"

    # Initialize list to hold all sides of the trigger
    trigger_sides = []

    # Function to create a side
    def create_side(v1, v2, v3, v4, material="TOOLS/TOOLSTRIGGER", uaxis="[0 0 1 0] 0.25", vaxis="[0 -1 0 0] 0.25"):
        nonlocal face_id_counter
        plane_str = f"({v1.x} {v1.y} {v1.z}) ({v2.x} {v2.y} {v2.z}) ({v3.x} {v3.y} {v3.z})"
        vertices_plus_data = {'v': [f"{v.x} {v.y} {v.z}" for v in [v1, v2, v3, v4]]}
        side_data = {
            'id': f'{face_id_counter}',
            'plane': plane_str,
            'vertices_plus': vertices_plus_data,
            'material': material,
            'uaxis': uaxis,
            'vaxis': vaxis,
            'rotation': '0',
            'lightmapscale': '16',
            'smoothing_groups': '0'
        }
        face_id_counter += 1
        return side_data

    # Create side faces (four sides)
    for i in range(4):
        v1_base = base_vertices[i]
        v2_base = base_vertices[(i + 1) % 4]
        v1_top = top_vertices[i]
        v2_top = top_vertices[(i + 1) % 4]

        # Define the side face with proper winding order
        side_face = create_side(v1_top, v2_top, v2_base, v1_base)
        trigger_sides.append(side_face)

    # Create bottom face
    bottom_face = create_side(base_vertices[0], base_vertices[1], base_vertices[2], base_vertices[3])
    trigger_sides.append(bottom_face)

    # Create top face (reversed order for correct normal)
    top_face = create_side(top_vertices[2], top_vertices[1], top_vertices[0], top_vertices[3])
    trigger_sides.append(top_face)

    # Build the solid
    trigger_solid = {
        'id': f'{solid_id_counter}',
        'side': trigger_sides,
        'editor': {
            'color': '220 30 220',
            'visgroupshown': '1',
            'visgroupautoshown': '1',
            'logicalpos': '[0 0]'
        }
    }

    # Build the entity
    trigger_entity = {
        'id': f'{solid_id_counter + 1}',
        'classname': 'trigger_multiple',
        'origin': origin,
        'spawnflags': '4097',  # Example spawnflags; adjust as needed
        'StartDisabled': '0',
        'solid': trigger_solid,
        'editor': {
            'color': '220 30 220',
            'visgroupshown': '1',
            'visgroupautoshown': '1'
        }
    }

    # Optionally add 'targetname' and 'wait' if provided
    if targetname:
        trigger_entity['targetname'] = targetname
    if wait:
        trigger_entity['wait'] = str(wait)

    return trigger_entity, face_id_counter

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("Automatic Trigger Multiple Generator")
    print("------------------------------------")
    print("This script generates trigger_multiple entities on faces with specified materials, including those in func_detail entities.")
    print("Output will be saved as 'generated_triggers.vmf'")
    print("------------------------------------")

    try:
        input_file = sys.argv[1]
        logging.info(f"Selected map: {input_file}")
    except IndexError:
        logging.error("No map selected. Please provide a .vmf file as a command-line argument.")
        sleep(4)
        return

    # Get user input for trigger height
    trigger_height = 4  # Default height
    try:
        trigger_height_input = input("Trigger height (default 4 units): ")
        if trigger_height_input.strip() != '':
            trigger_height = float(trigger_height_input)
        logging.info(f"Using trigger height: {trigger_height} units")
    except ValueError:
        logging.warning("Invalid input, using default height of 4 units.")

    # Get user input for materials
    materials_input = input("Enter materials to generate triggers for (comma-separated): ")
    materials_list = [mat.strip().lower() for mat in materials_input.split(',') if mat.strip()]
    if not materials_list:
        logging.error("No materials specified. Please provide at least one material.")
        sleep(4)
        return
    logging.info(f"Materials to target: {materials_list}")

    # Optional: Get user input for 'targetname' and 'wait'
    targetname_input = input("Enter targetname for triggers (leave blank for none): ").strip()
    wait_input = input("Enter wait time for triggers in seconds (default 1): ").strip()
    if wait_input:
        try:
            wait_time = float(wait_input)
        except ValueError:
            logging.warning("Invalid wait time input, using default value of 1 second.")
            wait_time = 1
    else:
        wait_time = 1  # Default wait time
    if wait_input:
        logging.info(f"Using wait time: {wait_time} seconds")
    else:
        logging.info(f"Using default wait time: {wait_time} seconds")

    parser = VMFParser()
    try:
        parser.parse(input_file)
    except Exception as e:
        logging.error(f"Error parsing VMF file: {e}")
        sleep(4)
        return

    # Collect new entities
    new_entities = []
    existing_solid_ids = [int(solid.id) for solid in parser.solids if solid.id and solid.id.isdigit()]
    existing_side_ids = [int(side.id) for solid in parser.solids for side in solid.sides if side.id and side.id.isdigit()]
    solid_id_counter = max(existing_solid_ids, default=10000) + 1
    face_id_counter = max(existing_side_ids, default=20000) + 1

    logging.info(f"Starting solid ID counter: {solid_id_counter}")
    logging.info(f"Starting face ID counter: {face_id_counter}")

    # Process solids
    solids_to_process = parser.solids

    for solid in solids_to_process:
        for side in solid.sides:
            try:
                # Check if the side's material matches the specified materials
                if side.material and side.material.lower() in materials_list:
                    # Generate trigger_multiple entity
                    trigger_entity, face_id_counter = generate_trigger_multiple(
                        solid_id_counter, face_id_counter, side, trigger_height, targetname=targetname_input if targetname_input else None, wait=wait_time)
                    if trigger_entity:
                        solid_id_counter += 2  # Increment for next solid and entity IDs
                        new_entities.append(trigger_entity)
                        logging.info(f"Generated trigger_multiple for side {side.id} of solid {solid.id}")
                    else:
                        logging.warning(f"Skipped side {side.id} of solid {solid.id} due to invalid geometry.")
            except Exception as e:
                logging.error(f"Error processing side {side.id} of solid {solid.id}: {e}")

    logging.info(f"Generated {len(new_entities)} new trigger_multiple entities")

    # Now, write the output VMF file
    output_file = "generated_triggers.vmf"
    try:
        with open(output_file, 'w') as f:
            # Write versioninfo, viewsettings, etc.
            def write_block(name, content, indent=0):
                ind = '    ' * indent
                f.write(f'{ind}{name}\n{ind}{{\n')
                for key, value in content.items():
                    if isinstance(value, dict):
                        write_block(key, value, indent + 1)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                write_block(key, item, indent + 1)
                            else:
                                f.write(f'{ind}    "{key}" "{item}"\n')
                    else:
                        f.write(f'{ind}    "{key}" "{value}"\n')
                f.write(f'{ind}}}\n')

            write_block('versioninfo', parser.versioninfo)
            f.write('\n')
            write_block('viewsettings', parser.viewsettings)
            f.write('\n')

            # Write world (excluding existing solids)
            f.write('world\n{\n')
            for key, value in parser.world.items():
                if key == 'solid':
                    continue
                if isinstance(value, dict):
                    write_block(key, value, indent=1)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            write_block(key, item, indent=1)
                        else:
                            f.write(f'    "{key}" "{item}"\n')
                else:
                    f.write(f'    "{key}" "{value}"\n')

            # Write existing world solids
            world_solids = parser.world.get('solid', [])
            if not isinstance(world_solids, list):
                world_solids = [world_solids]
            for solid_data in world_solids:
                write_block('solid', solid_data, indent=1)
            f.write('}\n\n')

            # Write existing entities
            for entity in parser.entities:
                write_block('entity', entity)
                f.write('\n')

            # Write new entities
            for entity in new_entities:
                write_block('entity', entity)
                f.write('\n')

            # Write cameras
            write_block('cameras', parser.cameras)
            f.write('\n')

            # Write cordons
            write_block('cordons', parser.cordons)
            f.write('\n')
    except Exception as e:
        logging.error(f"Error writing VMF file: {e}")
        sleep(4)
        return

    logging.info(f"Done. Generated triggers are saved in '{output_file}'")
    sleep(2)

if __name__ == '__main__':
    main()
