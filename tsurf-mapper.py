# Unfinished

import sys
import os
import logging
import math
import re
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QOpenGLWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QFileDialog, QMessageBox, QListWidget, QLabel,
    QFormLayout, QComboBox, QInputDialog, QSplitter, QProgressBar, QListWidgetItem,
    QAbstractItemView
)
from PyQt5.QtCore import Qt, QPoint, QObject, pyqtSignal
from PyQt5.QtGui import QCursor

from OpenGL.GL import *
from OpenGL.GLU import *

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
TRIGGER_EXTENSION_DISTANCE = 4.0  # Extension distance for triggers

# Utility Functions
def parse_vector(s):
    """Parses a string representing a vector into a tuple of floats."""
    try:
        return tuple(float(coord) for coord in s.strip().split())
    except ValueError as e:
        logging.error(f"Error parsing vector '{s}': {e}")
        return (0.0, 0.0, 0.0)

def parse_plane(s):
    """Parses a plane string into three Vertex instances."""
    try:
        points = re.findall(r'\((-?\d+\.?\d*) (-?\d+\.?\d*) (-?\d+\.?\d*)\)', s)
        if len(points) != 3:
            raise ValueError("Plane must have exactly three points.")
        return [Vertex(*point) for point in points]
    except Exception as e:
        logging.error(f"Error parsing plane '{s}': {e}")
        return [Vertex(0,0,0) for _ in range(3)]

# Data Structures
class Vertex:
    """Represents a 3D point or vector."""
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

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

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def scale(self, scalar):
        return Vertex(self.x * scalar, self.y * scalar, self.z * scalar)

    def normalize(self):
        mag = self.magnitude()
        if mag == 0:
            return Vertex(0, 0, 0)
        return Vertex(self.x / mag, self.y / mag, self.z / mag)

    def magnitude(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def as_tuple(self):
        return (self.x, self.y, self.z)

    def __repr__(self):
        return f"Vertex({self.x}, {self.y}, {self.z})"

class Face:
    """Represents a face of a solid."""
    def __init__(self, vertices, solid_id, side_id, material, uaxis, vaxis, rotation, lightmapscale, smoothing_groups):
        self.vertices = vertices  # List of Vertex instances
        self.solid_id = solid_id
        self.side_id = side_id
        self.material = material
        self.uaxis = uaxis
        self.vaxis = vaxis
        self.rotation = rotation
        self.lightmapscale = lightmapscale
        self.smoothing_groups = smoothing_groups
        self.selected = False

    def calculate_normal(self):
        """Calculates and returns the normal vector of the face."""
        if len(self.vertices) < 3:
            return Vertex(0, 0, 0)
        v1 = self.vertices[1] - self.vertices[0]
        v2 = self.vertices[2] - self.vertices[0]
        normal = v1.cross(v2).normalize()
        return normal

class Solid:
    """Represents a solid object in the VMF."""
    def __init__(self, solid_id):
        self.id = solid_id
        self.sides = []
        self.editor = {}  # Editor properties

class Side:
    """Represents a side (face) of a solid."""
    def __init__(self, side_id, plane_str, material, vertices_plus, uaxis, vaxis, rotation, lightmapscale, smoothing_groups):
        self.id = side_id
        self.plane_str = plane_str
        self.material = material
        self.vertices_plus = vertices_plus  # List of Vertex instances
        self.uaxis = uaxis
        self.vaxis = vaxis
        self.rotation = rotation
        self.lightmapscale = lightmapscale
        self.smoothing_groups = smoothing_groups

class Entity:
    """Represents an entity within the VMF."""
    def __init__(self, entity_id, classname, spawnflags, origin, properties=None, connections=None, editor=None, solids=None, hidden=None):
        self.id = entity_id
        self.classname = classname
        self.spawnflags = spawnflags
        self.origin = origin  # Vertex
        self.properties = properties if properties else {}
        self.connections = connections if connections else {}
        self.editor = editor if editor else {}
        self.solids = solids if solids else []
        self.hidden = hidden if hidden else {}

class VersionInfo:
    """Represents the versioninfo section of the VMF."""
    def __init__(self, editorversion, editorbuild, mapversion, formatversion, prefab):
        self.editorversion = int(editorversion)
        self.editorbuild = int(editorbuild)
        self.mapversion = int(mapversion)
        self.formatversion = int(formatversion)
        self.prefab = bool(int(prefab))

class VisGroup:
    """Represents a visgroup."""
    def __init__(self, name, visgroupid, color):
        self.name = name
        self.visgroupid = int(visgroupid)
        self.color = tuple(map(int, color.split()))

class ViewSettings:
    """Represents the viewsettings section."""
    def __init__(self, bSnapToGrid, bShowGrid, bShowLogicalGrid, nGridSpacing, bShow3DGrid):
        self.bSnapToGrid = bool(int(bSnapToGrid))
        self.bShowGrid = bool(int(bShowGrid))
        self.bShowLogicalGrid = bool(int(bShowLogicalGrid))
        self.nGridSpacing = int(nGridSpacing)
        self.bShow3DGrid = bool(int(bShow3DGrid))

class PalettePlus:
    """Represents the palette_plus section."""
    def __init__(self):
        self.colors = {}  # key: color index, value: (R, G, B)

    def add_color(self, index, rgb_str):
        try:
            rgb = tuple(map(int, rgb_str.strip().split()))
            self.colors[index] = rgb
            logging.debug(f"Added color{index}: {rgb}")
        except ValueError as e:
            logging.error(f"Error parsing color{index}: {e}")

class ColorCorrectionPlus:
    """Represents the colorcorrection_plus section."""
    def __init__(self):
        self.corrections = {}  # key: correction index, value: {'name': str, 'weight': float}

    def add_correction(self, index, name, weight):
        try:
            self.corrections[index] = {'name': name, 'weight': float(weight)}
            logging.debug(f"Added color correction{name}: weight={weight}")
        except ValueError as e:
            logging.error(f"Error parsing color correction{name}: {e}")

class LightPlus:
    """Represents the light_plus section."""
    def __init__(self):
        self.settings = {}

    def set_setting(self, key, value):
        self.settings[key] = self.convert_value(key, value)
        logging.debug(f"Set light_plus {key}: {self.settings[key]}")

    @staticmethod
    def convert_value(key, value):
        """Converts string values to appropriate data types."""
        int_keys = [
            "samples_sun", "samples_ambient", "samples_vis",
            "incremental_delay", "bake_dist", "ao_scale"
        ]
        float_keys = [
            "radius_scale", "brightness_scale"
        ]
        bool_keys = [
            "bounced", "incremental", "supersample",
            "bleed_hack", "soften_cosine", "debug",
            "cubemap", "hdr"
        ]
        if key in int_keys:
            return int(value)
        elif key in float_keys:
            return float(value)
        elif key in bool_keys:
            return bool(int(value))
        else:
            return value

class PostProcessPlus:
    """Represents the postprocess_plus section."""
    def __init__(self):
        self.settings = {}

    def set_setting(self, key, value):
        self.settings[key] = self.convert_value(key, value)
        logging.debug(f"Set postprocess_plus {key}: {self.settings[key]}")

    @staticmethod
    def convert_value(key, value):
        """Converts string values to appropriate data types."""
        float_keys = [
            "bloom_scale", "bloom_exponent", "bloom_saturation",
            "auto_exposure_min", "auto_exposure_max",
            "tonemap_percent_target", "tonemap_percent_bright_pixels",
            "tonemap_min_avg_luminance", "tonemap_rate",
            "vignette_start", "vignette_end", "vignette_blur"
        ]
        bool_keys = ["enable", "copied_from_controller", "vignette_enable"]
        if key in float_keys:
            return float(value)
        elif key in bool_keys:
            return bool(int(value))
        else:
            return value

class Camera:
    """Represents a single camera in the VMF."""
    def __init__(self, position=(0.0, 0.0, 0.0), target=(0.0, 0.0, 0.0), fov=60.0, near=0.1, far=10000.0):
        """
        Initializes a Camera instance.
        
        Args:
            position (tuple of float): The position coordinates of the camera.
            target (tuple of float): The look-at coordinates of the camera.
            fov (float): Field of view in degrees.
            near (float): Near clipping plane distance.
            far (float): Far clipping plane distance.
        """
        self.position = np.array(position, dtype=float)
        self.target = np.array(target, dtype=float)
        self.up = np.array([0.0, 1.0, 0.0], dtype=float)
        self.fov = fov
        self.near = near
        self.far = far

    def pan(self, dx, dy):
        """Pans the camera based on delta x and y."""
        # Implement panning logic
        right = np.cross(self.target - self.position, self.up)
        right = right / np.linalg.norm(right)
        up = self.up
        pan_speed = 0.01
        self.position += right * dx * pan_speed + up * dy * pan_speed
        self.target += right * dx * pan_speed + up * dy * pan_speed

    def orbit(self, dx, dy):
        """Orbits the camera around the target based on delta x and y."""
        # Implement orbiting logic
        orbit_speed = 0.005
        direction = self.position - self.target
        radius = np.linalg.norm(direction)
        theta = math.atan2(direction[1], direction[0])
        phi = math.acos(direction[2] / radius)
        theta += dx * orbit_speed
        phi -= dy * orbit_speed
        phi = max(0.1, min(math.pi - 0.1, phi))  # Prevent flipping
        direction = np.array([
            radius * math.sin(phi) * math.cos(theta),
            radius * math.sin(phi) * math.sin(theta),
            radius * math.cos(phi)
        ])
        self.position = self.target + direction

    def zoom(self, delta):
        """Zooms the camera in or out based on delta."""
        zoom_speed = 0.1
        direction = self.target - self.position
        distance = np.linalg.norm(direction)
        direction = direction / distance
        new_distance = distance - delta * zoom_speed
        new_distance = max(1.0, new_distance)  # Prevent negative or zero distance
        self.position = self.target - direction * new_distance

    def __repr__(self):
        return f"Camera(Position={self.position.tolist()}, Target={self.target.tolist()}, FOV={self.fov})"

class Cordon:
    """Represents a cordon in the VMF."""
    def __init__(self, active=False, areas=None):
        """
        Initializes a Cordon instance.
        
        Args:
            active (bool): Whether the cordon is active.
            areas (list of dict): List of areas defining the cordon.
        """
        self.active = active
        self.areas = areas if areas else []

    def __repr__(self):
        return f"Cordon(Active={self.active}, Areas={self.areas})"

class Overlay:
    """Represents an overlay in the VMF."""
    def __init__(self, name, properties=None):
        """
        Initializes an Overlay instance.
        
        Args:
            name (str): The name of the overlay.
            properties (dict): Additional properties of the overlay.
        """
        self.name = name
        self.properties = properties if properties else {}

    def __repr__(self):
        return f"Overlay(Name={self.name}, Properties={self.properties})"

class BgImagesPlus:
    """Represents the bgimages_plus section."""
    def __init__(self):
        self.images = []  # List of background images with their properties

    def add_image(self, image_properties):
        self.images.append(image_properties)
        logging.debug(f"Added background image: {image_properties}")

class VMFParser:
    """Parses VMF files and extracts solids, entities, and additional sections."""
    def __init__(self, progress_emitter=None):
        self.versioninfo = None
        self.visgroups = []
        self.viewsettings = None
        self.palette_plus = PalettePlus()
        self.colorcorrection_plus = ColorCorrectionPlus()
        self.light_plus = LightPlus()
        self.postprocess_plus = PostProcessPlus()
        self.bgimages_plus = BgImagesPlus()
        self.world = None
        self.entities = []
        self.cameras = []
        self.cordons = None
        self.overlay = []
        self.progress_emitter = progress_emitter

    def parse(self, file_path):
        """Parses the VMF file."""
        if not os.path.exists(file_path):
            logging.error(f"VMF file does not exist: {file_path}")
            return

        with open(file_path, 'r') as f:
            content = f.read()

        tokens = self.tokenize(content)
        token_iter = iter(tokens)
        parsed_data = self.parse_block(token_iter, parent=None)
        self.process_parsed_data(parsed_data)

    def parse_content(self, content):
        """Parses VMF content from a string."""
        tokens = self.tokenize(content)
        token_iter = iter(tokens)
        parsed_data = self.parse_block(token_iter, parent=None)
        self.process_parsed_data(parsed_data)

    def tokenize(self, content):
        """Tokenizes the VMF content."""
        # Regex to match quoted strings, braces, brackets, or other tokens
        token_pattern = r'"[^"]*"|\{|\}|\[|\]|[^"\s{}\[\]]+'
        tokens = re.findall(token_pattern, content)
        tokens = [token.strip() for token in tokens if token.strip()]
        logging.debug(f"Tokenized content into {len(tokens)} tokens.")
        return tokens

    def parse_block(self, tokens, parent):
        """Recursively parses a block of tokens."""
        current = {}
        try:
            while True:
                token = next(tokens)
                logging.debug(f"Parsing token: {token} (Parent: {parent})")
                if token == '}':
                    return current
                elif token.startswith('"') and token.endswith('"'):
                    key = token.strip('"')
                    next_token = next(tokens)
                    if next_token == '{':
                        # Nested block
                        value = self.parse_block(tokens, parent=key)
                        if key not in current:
                            current[key] = value
                        else:
                            if isinstance(current[key], list):
                                current[key].append(value)
                            else:
                                current[key] = [current[key], value]
                    elif next_token.startswith('"') and next_token.endswith('"'):
                        # Handle key-value pairs
                        value = next_token.strip('"')
                        if key in current:
                            if isinstance(current[key], list):
                                current[key].append(value)
                            else:
                                current[key] = [current[key], value]
                        else:
                            current[key] = value
                    else:
                        # Handle non-quoted values
                        value = next_token
                        if key in current:
                            if isinstance(current[key], list):
                                current[key].append(value)
                            else:
                                current[key] = [current[key], value]
                        else:
                            current[key] = value
                elif token == '{':
                    # Handle anonymous or unexpected blocks
                    logging.warning(f"Unexpected '{{' encountered without a key at position. Skipping block.")
                    self.skip_block(tokens)
                else:
                    # Handle other tokens 
                    logging.warning(f"Unexpected token: {token}. Skipping.")
        except StopIteration:
            return current

    def skip_block(self, tokens):
        """Skips tokens until the matching closing brace is found."""
        depth = 1
        try:
            while depth > 0:
                token = next(tokens)
                logging.debug(f"Skipping token during block skip: {token}")
                if token == '{':
                    depth += 1
                elif token == '}':
                    depth -= 1
        except StopIteration:
            logging.error("Reached end of tokens while skipping a block.")

    def process_parsed_data(self, data):
        """Processes the parsed data into structured objects."""
        # Process versioninfo
        if 'versioninfo' in data:
            vi = data['versioninfo']
            self.versioninfo = VersionInfo(
                editorversion=vi.get('editorversion', '400'),
                editorbuild=vi.get('editorbuild', '8866'),
                mapversion=vi.get('mapversion', '435'),
                formatversion=vi.get('formatversion', '100'),
                prefab=vi.get('prefab', '0')
            )
            logging.debug(f"Parsed VersionInfo: {self.versioninfo.__dict__}")

        # Process visgroups
        if 'visgroups' in data:
            vg = data['visgroups']
            visgroup_entries = vg.get('visgroup', [])
            if not isinstance(visgroup_entries, list):
                visgroup_entries = [visgroup_entries]
            for vg_entry in visgroup_entries:
                name = vg_entry.get('name', 'Unnamed')
                visgroupid = vg_entry.get('visgroupid', '0')
                color = vg_entry.get('color', '255 255 255')
                visgroup = VisGroup(name, visgroupid, color)
                self.visgroups.append(visgroup)
            logging.debug(f"Parsed {len(self.visgroups)} VisGroups.")

        # Process viewsettings
        if 'viewsettings' in data:
            vs = data['viewsettings']
            self.viewsettings = ViewSettings(
                bSnapToGrid=vs.get('bSnapToGrid', '1'),
                bShowGrid=vs.get('bShowGrid', '1'),
                bShowLogicalGrid=vs.get('bShowLogicalGrid', '0'),
                nGridSpacing=vs.get('nGridSpacing', '1'),
                bShow3DGrid=vs.get('bShow3DGrid', '0')
            )
            logging.debug(f"Parsed ViewSettings: {self.viewsettings.__dict__}")

        # Process palette_plus
        if 'palette_plus' in data:
            pp = data['palette_plus']
            for key, value in pp.items():
                if key.startswith('color'):
                    try:
                        index = int(key.replace('color', ''))
                        self.palette_plus.add_color(index, value)
                    except ValueError as e:
                        logging.error(f"Error parsing palette_plus key '{key}': {e}")
            logging.debug(f"Parsed PalettePlus with {len(self.palette_plus.colors)} colors.")

        # Process colorcorrection_plus
        if 'colorcorrection_plus' in data:
            cc = data['colorcorrection_plus']
            for key in cc:
                if key.startswith('name'):
                    try:
                        index = int(key.replace('name', ''))
                        name = cc.get(f"name{index}", "")
                        weight = cc.get(f"weight{index}", "1")
                        self.colorcorrection_plus.add_correction(index, name, weight)
                    except ValueError as e:
                        logging.error(f"Error parsing colorcorrection_plus key '{key}': {e}")
            logging.debug(f"Parsed ColorCorrectionPlus with {len(self.colorcorrection_plus.corrections)} corrections.")

        # Process light_plus
        if 'light_plus' in data:
            lp = data['light_plus']
            for key, value in lp.items():
                self.light_plus.set_setting(key, value)
            logging.debug(f"Parsed LightPlus settings: {self.light_plus.settings}")

        # Process postprocess_plus
        if 'postprocess_plus' in data:
            pp_plus = data['postprocess_plus']
            for key, value in pp_plus.items():
                self.postprocess_plus.set_setting(key, value)
            logging.debug(f"Parsed PostProcessPlus settings: {self.postprocess_plus.settings}")

        # Process bgimages_plus
        if 'bgimages_plus' in data:
            bg = data['bgimages_plus']
            # Will add this at some point
            logging.debug("Parsed BgImagesPlus with images: Not implemented.")

        # Process world
        if 'world' in data:
            world_data = data['world']
            self.world = self.parse_world(world_data)

        # Process entities
        if 'entity' in data:
            entities_data = data['entity']
            if not isinstance(entities_data, list):
                entities_data = [entities_data]
            for ent_data in entities_data:
                entity = self.parse_entity(ent_data)
                self.entities.append(entity)
            logging.debug(f"Parsed {len(self.entities)} Entities.")

        # Process cameras
        if 'cameras' in data:
            cameras_data = data['cameras']
            active_camera_id = int(cameras_data.get('activecamera', '0'))
            camera_entries = cameras_data.get('camera', [])
            if not isinstance(camera_entries, list):
                camera_entries = [camera_entries]
            for idx, cam_data in enumerate(camera_entries):
                position_str = cam_data.get('position', '[0 0 0]')
                look_str = cam_data.get('look', '[0 0 0]')
                position = parse_vector(position_str.strip('[]'))
                look = parse_vector(look_str.strip('[]'))
                camera = Camera(position=position, target=look)
                self.cameras.append(camera)
                logging.debug(f"Parsed Camera {idx}: {camera}")
            logging.debug(f"Parsed {len(self.cameras)} Cameras. Active Camera ID: {active_camera_id}")

        # Process cordons
        if 'cordons' in data:
            cordons_data = data['cordons']
            active = bool(int(cordons_data.get('active', '0')))
            # Unfinished
            cordon = Cordon(active=active, areas=cordons_data.get('area', []))
            self.cordons = cordon
            logging.debug(f"Parsed Cordon: {self.cordons}")

        # Process overlay
        if 'overlay' in data:
            overlay_data = data['overlay']
            overlay_entries = overlay_data.get('overlay', [])
            if not isinstance(overlay_entries, list):
                overlay_entries = [overlay_entries]
            for ov_data in overlay_entries:
                name = ov_data.get('name', 'Unnamed')
                properties = {}
                for key, value in ov_data.items():
                    if key != 'name':
                        properties[key] = value
                overlay = Overlay(name=name, properties=properties)
                self.overlay.append(overlay)
                logging.debug(f"Parsed Overlay: {overlay}")
            logging.debug(f"Parsed {len(self.overlay)} Overlays.")

        # Additional processing for other sections can be added here.

    def parse_world(self, world_data):
        """Parses the world section."""
        world_id = int(world_data.get('id', '1'))
        mapversion = world_data.get('mapversion', '1')
        classname = world_data.get('classname', 'worldspawn')
        skyname = world_data.get('skyname', 'sky_dust')

        world = Entity(
            entity_id=world_id,
            classname=classname,
            spawnflags=0,
            origin=Vertex(0, 0, 0),
            properties={
                'mapversion': mapversion,
                'skyname': skyname
            },
            solids=[]
        )

        # Parse Solids
        solids_data = world_data.get('solid', [])
        if not isinstance(solids_data, list):
            solids_data = [solids_data]
        for solid_data in solids_data:
            solid = self.parse_solid(solid_data)
            world.solids.append(solid)
        
        logging.debug(f"Parsed World with ID {world_id} containing {len(world.solids)} solids.")
        return world

    def parse_solid(self, solid_data):
        """Parses a solid section."""
        solid_id = int(solid_data.get('id', '-1'))
        solid = Solid(solid_id=solid_id)

        # Parse sides
        sides_data = solid_data.get('side', [])
        if not isinstance(sides_data, list):
            sides_data = [sides_data]
        for side_data in sides_data:
            side = self.parse_side(side_data)
            solid.sides.append(side)
        
        # Parse editor
        editor_data = solid_data.get('editor', {})
        solid.editor = editor_data

        logging.debug(f"Parsed Solid ID {solid_id} with {len(solid.sides)} sides.")
        return solid

    def parse_side(self, side_data):
        """Parses a side section."""
        side_id = int(side_data.get('id', '-1'))
        plane_str = side_data.get('plane', '(0 0 0) (0 0 0) (0 0 0)')
        material = side_data.get('material', 'TOOLS/TOOLSNODRAW')
        uaxis = side_data.get('uaxis', '[1 0 0 0] 0.25')
        vaxis = side_data.get('vaxis', '[0 -1 0 0] 0.25')
        rotation = float(side_data.get('rotation', '0'))
        lightmapscale = float(side_data.get('lightmapscale', '16'))
        smoothing_groups = int(side_data.get('smoothing_groups', '0'))

        # Parse vertices_plus
        vertices_plus_data = side_data.get('vertices_plus', {}).get('v', [])
        if not isinstance(vertices_plus_data, list):
            vertices_plus_data = [vertices_plus_data]
        vertices = [Vertex(*parse_vector(v_str)) for v_str in vertices_plus_data]

        side = Side(
            side_id=side_id,
            plane_str=plane_str,
            material=material,
            vertices_plus=vertices,
            uaxis=uaxis,
            vaxis=vaxis,
            rotation=rotation,
            lightmapscale=lightmapscale,
            smoothing_groups=smoothing_groups
        )

        logging.debug(f"Parsed Side ID {side_id} with Material '{material}'.")
        return side

    def parse_entity(self, ent_data):
        """Parses an entity section."""
        entity_id = int(ent_data.get('id', '-1'))
        classname = ent_data.get('classname', 'info_null')
        spawnflags = int(ent_data.get('spawnflags', '0'))
        origin_str = ent_data.get('origin', '0 0 0')
        origin = Vertex(*parse_vector(origin_str))

        # Parse properties
        properties = {}
        for key, value in ent_data.items():
            if key not in ['id', 'classname', 'spawnflags', 'origin', 'connections', 'editor', 'solid', 'hidden']:
                properties[key] = value

        # Parse connections
        connections_data = ent_data.get('connections', {})
        connections = {}
        for key, value in connections_data.items():
            connections[key] = value

        # Parse editor
        editor_data = ent_data.get('editor', {})

        # Parse solids (if any)
        solids = []
        solids_data = ent_data.get('solid', [])
        if not isinstance(solids_data, list):
            solids_data = [solids_data]
        for solid_data in solids_data:
            solid = self.parse_solid(solid_data)
            solids.append(solid)

        # Parse hidden (if any)
        hidden_data = ent_data.get('hidden', {})

        entity = Entity(
            entity_id=entity_id,
            classname=classname,
            spawnflags=spawnflags,
            origin=origin,
            properties=properties,
            connections=connections,
            editor=editor_data,
            solids=solids,
            hidden=hidden_data
        )

        logging.debug(f"Parsed Entity ID {entity_id} with classname '{classname}'.")
        return entity

    def generate_unique_id(self):
        """Generates a unique ID for new entities or solids."""
        existing_ids = set()
        if self.world:
            existing_ids.add(self.world.id)
            for solid in self.world.solids:
                existing_ids.add(solid.id)
                for side in solid.sides:
                    existing_ids.add(side.id)
        for entity in self.entities:
            existing_ids.add(entity.id)
            for solid in entity.solids:
                existing_ids.add(solid.id)
                for side in solid.sides:
                    existing_ids.add(side.id)
        unique_id = 1
        while unique_id in existing_ids:
            unique_id += 1
        return unique_id

class MapViewport(QOpenGLWidget):
    """The OpenGL viewport for rendering the map and triggers."""
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window  # Reference to MainWindow
        self.map_data = None
        self.faces = []
        self.triggers = []
        self.selected_faces = set()
        self.selected_trigger = None
        self.camera = Camera()
        self.last_pos = QPoint()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.multi_select_mode = False

        # Initialize matrices and viewport data
        self.modelview_matrix = []
        self.projection_matrix = []
        self.viewport_data = []

    def initializeGL(self):
        """Initializes OpenGL settings."""
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)  # Enable face culling
        glShadeModel(GL_SMOOTH)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        # Set up light parameters
        glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 0.0, 1000.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])

    def resizeGL(self, width, height):
        """Handles window resizing."""
        glViewport(0, 0, width, height)
        self.update_projection()

    def update_projection(self):
        """Updates the projection matrix."""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = self.width() / self.height() if self.height() != 0 else 1.0
        gluPerspective(self.camera.fov, aspect, self.camera.near, self.camera.far)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(*self.camera.position, *self.camera.target, *self.camera.up)

    def paintGL(self):
        """Renders the scene."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(*self.camera.position, *self.camera.target, *self.camera.up)
        self.render_map()
        self.render_triggers()

        # Capture the matrices after setting up the view
        self.modelview_matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
        self.projection_matrix = glGetDoublev(GL_PROJECTION_MATRIX)
        self.viewport_data = glGetIntegerv(GL_VIEWPORT)

    def render_map(self):
        """Renders the map geometry."""
        glDisable(GL_LIGHTING)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)
        for face in self.faces:
            if face.selected:
                glColor3f(1.0, 0.0, 0.0)  # Red for selected faces
            else:
                glColor3f(0.8, 0.8, 0.8)  # Grey for unselected faces
            glBegin(GL_POLYGON)
            for vertex in face.vertices:
                glVertex3f(vertex.x, vertex.y, vertex.z)
            glEnd()
        glDisable(GL_POLYGON_OFFSET_FILL)
        glEnable(GL_LIGHTING)

    def render_triggers(self):
        """Renders the triggers."""
        glColor3f(0.0, 1.0, 0.0)  # Green color for triggers
        glLineWidth(2.0)
        for trigger in self.triggers:
            if isinstance(trigger, Entity):
                for solid in trigger.solids:
                    if isinstance(solid, Solid):
                        for side in solid.sides:
                            vertices = [v.as_tuple() for v in side.vertices_plus]
                            if len(vertices) >= 4:
                                glBegin(GL_QUADS)
                                for v in vertices:
                                    glVertex3f(*v)
                                glEnd()
            else:
                logging.warning(f"Invalid trigger object: {type(trigger)}")
        glLineWidth(1.0)

    def set_map_data(self, solids):
        """Sets the map data and extracts faces."""
        self.faces.clear()
        for solid in solids:
            solid_id = solid.id
            for side in solid.sides:
                vertices = side.vertices_plus
                if len(vertices) >= 3:
                    face = Face(vertices, solid_id, side.id, side.material,
                                side.uaxis, side.vaxis, side.rotation,
                                side.lightmapscale, side.smoothing_groups)
                    self.faces.append(face)
        self.update()

    def mousePressEvent(self, event):
        """Handles mouse press events."""
        self.last_pos = event.pos()
        if event.button() == Qt.LeftButton:
            modifiers = QApplication.keyboardModifiers()
            self.multi_select_mode = modifiers == Qt.ControlModifier
            self.select_face(event.pos())
        elif event.button() == Qt.MiddleButton:
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        """Handles mouse move events."""
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()

        if event.buttons() & Qt.MiddleButton:
            self.camera.pan(dx, -dy)
            self.update()
        elif event.buttons() & Qt.RightButton:
            self.camera.orbit(dx, dy)
            self.update()
        self.last_pos = event.pos()

    def mouseReleaseEvent(self, event):
        """Handles mouse release events."""
        if event.button() == Qt.MiddleButton:
            self.setCursor(Qt.ArrowCursor)

    def wheelEvent(self, event):
        """Handles mouse wheel events."""
        delta = event.angleDelta().y()
        self.camera.zoom(delta)
        self.update()

    def keyPressEvent(self, event):
        """Handles key press events."""
        if event.key() == Qt.Key_F:
            self.focus_on_selection()
        elif event.key() == Qt.Key_R:
            self.reset_camera()
        event.accept()

    def select_face(self, pos):
        """Selects a face based on the mouse position."""
        ray_origin, ray_direction = self.get_ray(pos)
        closest_face = None
        min_distance = float('inf')
        for face in self.faces:
            intersection = self.ray_face_intersection(ray_origin, ray_direction, face)
            if intersection is not None:
                distance = np.linalg.norm(intersection - ray_origin)
                if distance < min_distance:
                    min_distance = distance
                    closest_face = face
        if closest_face:
            face_id = (closest_face.solid_id, closest_face.side_id)
            if self.multi_select_mode:
                if face_id in self.selected_faces:
                    self.selected_faces.remove(face_id)
                    closest_face.selected = False
                else:
                    self.selected_faces.add(face_id)
                    closest_face.selected = True
            else:
                if not self.multi_select_mode:
                    for f in self.faces:
                        f.selected = False
                    self.selected_faces.clear()
                self.selected_faces.add(face_id)
                closest_face.selected = True
            self.update()

    def get_ray(self, pos):
        """Generates a ray from the camera through the viewport pixel."""
        try:
            x = pos.x()
            y = self.height() - pos.y()
            modelview = self.modelview_matrix.flatten().tolist()
            projection = self.projection_matrix.flatten().tolist()
            viewport = self.viewport_data.tolist()

            # Debugging: Print types and lengths
            logging.debug(f"ModelView Matrix Type: {type(modelview)}, Length: {len(modelview)}")
            logging.debug(f"Projection Matrix Type: {type(projection)}, Length: {len(projection)}")
            logging.debug(f"Viewport Type: {type(viewport)}, Length: {len(viewport)}")

            # Validate lengths
            assert len(modelview) == 16, "ModelView matrix should have 16 elements."
            assert len(projection) == 16, "Projection matrix should have 16 elements."
            assert len(viewport) == 4, "Viewport should have 4 elements."

            winX = float(x)
            winY = float(y)
            winZ = 0.0
            near = gluUnProject(winX, winY, winZ, modelview, projection, viewport)
            winZ = 1.0
            far = gluUnProject(winX, winY, winZ, modelview, projection, viewport)
            ray_origin = np.array(near)
            ray_direction = np.array(far) - np.array(near)
            ray_direction /= np.linalg.norm(ray_direction)
            return ray_origin, ray_direction
        except Exception as e:
            logging.error(f"Error in get_ray: {e}")
            return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])

    def ray_face_intersection(self, ray_origin, ray_direction, face):
        """
        Checks for intersection between a ray and a face.
        Returns the intersection point as a NumPy array if it exists, otherwise None.
        """
        if len(face.vertices) < 3:
            return None  # Not a valid polygon

        p0 = np.array(face.vertices[0].as_tuple())
        p1 = np.array(face.vertices[1].as_tuple())
        p2 = np.array(face.vertices[2].as_tuple())

        # Compute two edge vectors
        v1 = p1 - p0
        v2 = p2 - p0

        # Compute the plane normal
        normal = np.cross(v1, v2)
        norm_length = np.linalg.norm(normal)
        if norm_length == 0:
            return None  # Degenerate plane
        normal = normal / norm_length

        # Compute intersection of ray with plane
        denom = np.dot(normal, ray_direction)
        if abs(denom) < 1e-6:
            return None  # Parallel, no intersection

        d = np.dot(normal, p0)
        t = (d - np.dot(normal, ray_origin)) / denom
        if t < 0:
            return None  # Intersection behind the ray origin

        intersection_point = ray_origin + t * ray_direction

        # Check if the intersection point lies within the polygon
        if self.point_in_polygon(intersection_point, face):
            return intersection_point
        else:
            return None

    def point_in_polygon(self, point, face):
        """
        Determines if a point lies within a polygon (face) using the ray casting algorithm.
        The polygon is assumed to be convex and defined in 3D space.
        """
        # Project the polygon and the point onto the dominant plane
        normal = face.calculate_normal()
        abs_normal = np.abs([normal.x, normal.y, normal.z])
        dominant_axis = np.argmax(abs_normal)

        # Define projection indices based on dominant axis
        if dominant_axis == 0:
            i1, i2 = 1, 2  # Project to YZ
        elif dominant_axis == 1:
            i1, i2 = 0, 2  # Project to XZ
        else:
            i1, i2 = 0, 1  # Project to XY

        # Extract projected vertices
        vertices = [(v.as_tuple()[i1], v.as_tuple()[i2]) for v in face.vertices]
        x, y = point[i1], point[i2]

        # Ray casting algorithm
        inside = False
        n = len(vertices)
        for i in range(n):
            j = (i + 1) % n
            xi, yi = vertices[i]
            xj, yj = vertices[j]

            intersect = ((yi > y) != (yj > y)) and \
                        (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
            if intersect:
                inside = not inside
        return inside

    def focus_on_selection(self):
        """Focuses the camera on the selected faces."""
        if self.selected_faces:
            centers = []
            for face_id in self.selected_faces:
                face = self.main_window.get_face(*face_id)  # Accessing MainWindow's method
                if face:
                    center = np.mean([np.array([v.x, v.y, v.z]) for v in face.vertices], axis=0)
                    centers.append(center)
            if centers:
                average_center = np.mean(centers, axis=0)
                self.camera.target = average_center
                self.camera.position = average_center + np.array([0.0, 0.0, 1000.0])
                self.update()

    def reset_camera(self):
        """Resets the camera to the default position."""
        self.camera = Camera()
        self.update()

class Command:
    """Base class for commands."""
    def execute(self):
        pass

    def undo(self):
        pass

class AddTriggerCommand(Command):
    """Command to add triggers."""
    def __init__(self, main_window, targetname, selected_faces):
        self.main_window = main_window
        self.targetname = targetname
        self.selected_faces = selected_faces.copy()
        self.added_triggers = []

    def execute(self):
        for solid_id, side_id in self.selected_faces:
            face = self.main_window.get_face(solid_id, side_id)
            if face:
                trigger = self.main_window.generate_trigger(face, self.targetname)
                if trigger:
                    self.main_window.add_trigger(trigger)
                    self.added_triggers.append(trigger)
        self.main_window.update_trigger_list()

    def undo(self):
        for trigger in self.added_triggers:
            self.main_window.remove_trigger(trigger)
        self.main_window.update_trigger_list()

class MainWindow(QMainWindow):
    """The main window of the application."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VMF Trigger Generator")
        self.setGeometry(100, 100, 1200, 800)
        self.map_data = None
        self.solids = []
        self.undo_stack = []
        self.redo_stack = []
        self.setup_ui()
        self.progress_emitter = ProgressEmitter()
        self.progress_emitter.progressUpdated.connect(self.update_progress_bar)

    def setup_ui(self):
        """Sets up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout()
        central_widget.setLayout(layout)

        # Left panel (Viewport)
        self.viewport = MapViewport(main_window=self)
        layout.addWidget(self.viewport, 3)

        # Right panel (Controls)
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        self.targetname_input = QLineEdit()
        self.targetname_input.setPlaceholderText("Enter targetname for triggers")
        right_layout.addWidget(self.targetname_input)

        self.add_trigger_button = QPushButton("Add Trigger")
        self.add_trigger_button.clicked.connect(self.add_trigger)
        right_layout.addWidget(self.add_trigger_button)

        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo)
        self.undo_button.setEnabled(False)
        right_layout.addWidget(self.undo_button)

        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self.redo)
        self.redo_button.setEnabled(False)
        right_layout.addWidget(self.redo_button)

        self.open_file_button = QPushButton("Open VMF")
        self.open_file_button.clicked.connect(self.open_vmf_file)
        right_layout.addWidget(self.open_file_button)

        self.save_file_button = QPushButton("Save VMF")
        self.save_file_button.clicked.connect(self.save_vmf_file)
        right_layout.addWidget(self.save_file_button)

        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)

        # Trigger list
        self.trigger_list = QListWidget()
        self.trigger_list.itemSelectionChanged.connect(self.on_trigger_selection_changed)
        self.trigger_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        right_layout.addWidget(self.trigger_list)

        self.info_label = QLabel("No face or trigger selected")
        right_layout.addWidget(self.info_label)

        layout.addWidget(right_panel, 1)

    def open_vmf_file(self):
        """Opens a VMF file."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Open VMF File", "", "VMF Files (*.vmf)")
        if file_name:
            try:
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(0)

                parser = VMFParser(progress_emitter=self.progress_emitter)
                parser.parse(file_name)
                self.map_data = parser
                self.solids = parser.world.solids if parser.world else []
                self.viewport.set_map_data(self.solids)
                self.focus_on_map()
                self.update_trigger_list()

                self.progress_bar.setVisible(False)
                QMessageBox.information(self, "Success", f"Loaded VMF file: {file_name}")
                logging.info(f"Loaded VMF file: {file_name}")
            except Exception as e:
                self.progress_bar.setVisible(False)
                QMessageBox.critical(self, "Error", f"Error loading VMF file: {str(e)}")
                logging.error(f"Error loading VMF file: {str(e)}")

    def save_vmf_file(self):
        """Saves the VMF file with the added triggers."""
        if not self.map_data:
            QMessageBox.warning(self, "Warning", "No VMF file is currently loaded.")
            return
        file_name, _ = QFileDialog.getSaveFileName(self, "Save VMF File", "", "VMF Files (*.vmf)")
        if file_name:
            try:
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(0)

                total_steps = (len(self.map_data.solids) +
                               len(self.map_data.entities) +
                               len(self.viewport.triggers))
                current_step = 0

                with open(file_name, 'w') as f:
                    # Write versioninfo
                    f.write('versioninfo\n{\n')
                    vi = self.map_data.versioninfo
                    f.write(f'\t"editorversion" "{vi.editorversion}"\n')
                    f.write(f'\t"editorbuild" "{vi.editorbuild}"\n')
                    f.write(f'\t"mapversion" "{vi.mapversion}"\n')
                    f.write(f'\t"formatversion" "{vi.formatversion}"\n')
                    f.write(f'\t"prefab" "{int(vi.prefab)}"\n')
                    f.write('}\n\n')

                    # Write visgroups
                    f.write('visgroups\n{\n')
                    for vg in self.map_data.visgroups:
                        f.write('\tvisgroup\n\t{\n')
                        f.write(f'\t\t"name" "{vg.name}"\n')
                        f.write(f'\t\t"visgroupid" "{vg.visgroupid}"\n')
                        f.write(f'\t\t"color" "{vg.color[0]} {vg.color[1]} {vg.color[2]}"\n')
                        f.write('\t}\n')
                    f.write('}\n\n')

                    # Write viewsettings
                    f.write('viewsettings\n{\n')
                    vs = self.map_data.viewsettings
                    f.write(f'\t"bSnapToGrid" "{int(vs.bSnapToGrid)}"\n')
                    f.write(f'\t"bShowGrid" "{int(vs.bShowGrid)}"\n')
                    f.write(f'\t"bShowLogicalGrid" "{int(vs.bShowLogicalGrid)}"\n')
                    f.write(f'\t"nGridSpacing" "{vs.nGridSpacing}"\n')
                    f.write(f'\t"bShow3DGrid" "{int(vs.bShow3DGrid)}"\n')
                    f.write('}\n\n')

                    # Write world
                    if self.map_data.world:
                        f.write('world\n{\n')
                        f.write(f'\t"id" "{self.map_data.world.id}"\n')
                        f.write(f'\t"mapversion" "{self.map_data.world.properties.get("mapversion", "1")}"\n')
                        f.write(f'\t"classname" "{self.map_data.world.classname}"\n')
                        f.write(f'\t"skyname" "{self.map_data.world.properties.get("skyname", "sky_dust")}"\n')
                        for solid in self.map_data.world.solids:
                            self.write_solid(f, solid, indent_level=1)
                            current_step += 1
                            self.progress_emitter.progressUpdated.emit(int(current_step / total_steps * 100))
                        f.write('}\n\n')

                    # Write entities
                    for entity in self.map_data.entities:
                        self.write_entity(f, entity, indent_level=0)
                        current_step += 1
                        self.progress_emitter.progressUpdated.emit(int(current_step / total_steps * 100))

                    # Write triggers
                    for trigger in self.viewport.triggers:
                        self.write_entity(f, trigger, indent_level=0)
                        current_step += 1
                        self.progress_emitter.progressUpdated.emit(int(current_step / total_steps * 100))

                    # This is unfinished

                self.progress_bar.setVisible(False)
                QMessageBox.information(self, "Success", f"VMF file saved successfully: {file_name}")
                logging.info(f"VMF file saved successfully: {file_name}")
            except Exception as e:
                self.progress_bar.setVisible(False)
                QMessageBox.critical(self, "Error", f"Error saving VMF file: {str(e)}")
                logging.error(f"Error saving VMF file: {str(e)}")

    def write_solid(self, file, solid, indent_level=1):
        """Writes a solid to the VMF file."""
        indent = '\t' * indent_level
        file.write(f'{indent}solid\n{indent}{{\n')
        file.write(f'{indent}\t"id" "{solid.id}"\n')
        for side in solid.sides:
            file.write(f'{indent}\tside\n{indent}\t{{\n')
            file.write(f'{indent}\t\t"id" "{side.id}"\n')
            file.write(f'{indent}\t\t"plane" "{side.plane_str}"\n')
            file.write(f'{indent}\t\t"material" "{side.material}"\n')
            file.write(f'{indent}\t\t"uaxis" "{side.uaxis}"\n')
            file.write(f'{indent}\t\t"vaxis" "{side.vaxis}"\n')
            file.write(f'{indent}\t\t"rotation" "{side.rotation}"\n')
            file.write(f'{indent}\t\t"lightmapscale" "{side.lightmapscale}"\n')
            file.write(f'{indent}\t\t"smoothing_groups" "{side.smoothing_groups}"\n')
            # Handle vertices_plus
            file.write(f'{indent}\t\tvertices_plus\n{indent}\t\t{{\n')
            for vertex in side.vertices_plus:
                file.write(f'\t\t\t"v" "{vertex.x} {vertex.y} {vertex.z}"\n')
            file.write(f'{indent}\t\t}}\n')
            file.write(f'{indent}\t}}\n')
        # Handle editor properties
        if solid.editor:
            file.write(f'{indent}\teditor\n{indent}\t{{\n')
            for key, value in solid.editor.items():
                file.write(f'\t\t"{key}" "{value}"\n')
            file.write(f'{indent}\t}}\n')
        file.write(f'{indent}}}\n')

    def write_entity(self, file, entity, indent_level=0):
        """Writes an entity to the VMF file."""
        indent = '\t' * indent_level
        file.write(f'{indent}entity\n{indent}{{\n')
        file.write(f'\t"id" "{entity.id}"\n')
        file.write(f'\t"classname" "{entity.classname}"\n')
        file.write(f'\t"spawnflags" "{entity.spawnflags}"\n')
        file.write(f'\t"origin" "{entity.origin.x} {entity.origin.y} {entity.origin.z}"\n')
        # Write additional properties
        for key, value in entity.properties.items():
            file.write(f'\t"{key}" "{value}"\n')
        # Write connections
        if entity.connections:
            file.write(f'\tconnections\n\t{{\n')
            for output, target in entity.connections.items():
                file.write(f'\t\t"{output}" "{target}"\n')
            file.write(f'\t}}\n')
        # Write solids
        for solid in entity.solids:
            self.write_solid(file, solid, indent_level=1)
        # Write editor
        if entity.editor:
            file.write(f'\teditor\n\t{{\n')
            for key, value in entity.editor.items():
                file.write(f'\t\t"{key}" "{value}"\n')
            file.write(f'\t}}\n')
        file.write(f'{indent}}}\n\n')

    def add_trigger(self, trigger):
        """Adds a trigger to the viewport."""
        self.viewport.triggers.append(trigger)

    def remove_trigger(self, trigger):
        """Removes a trigger from the viewport."""
        self.viewport.triggers.remove(trigger)

    def update_trigger_list(self):
        """Updates the list of triggers in the UI."""
        self.trigger_list.clear()
        for trigger in self.viewport.triggers:
            item = QListWidgetItem(f"Trigger: {trigger.properties.get('targetname', 'Unnamed')}")
            item.setData(Qt.UserRole, trigger)
            self.trigger_list.addItem(item)

    def on_trigger_selection_changed(self):
        """Handles trigger selection in the list."""
        selected_items = self.trigger_list.selectedItems()
        self.selected_triggers = [item.data(Qt.UserRole) for item in selected_items]
        self.update_info_label()

    def update_info_label(self):
        """Updates the info label with details about the selected faces or triggers."""
        if hasattr(self, 'selected_triggers') and self.selected_triggers:
            trigger_info = f"Selected Triggers: {len(self.selected_triggers)}\n"
            if len(self.selected_triggers) == 1:
                trigger = self.selected_triggers[0]
                trigger_info += f"Targetname: {trigger.properties.get('targetname', 'Unnamed')}\n"
                trigger_info += f"Origin: {trigger.origin.as_tuple()}\n"
                trigger_info += f"Spawnflags: {trigger.spawnflags}"
            self.info_label.setText(trigger_info)
        elif self.viewport.selected_faces:
            face_count = len(self.viewport.selected_faces)
            face_info = f"Selected Faces: {face_count}\n"
            if face_count == 1:
                face = next(iter(self.viewport.selected_faces))
                face_info += f"Solid ID: {face[0]}, Side ID: {face[1]}"
            self.info_label.setText(face_info)
        else:
            self.info_label.setText("No face or trigger selected")

    def get_face(self, solid_id, side_id):
        """Gets a face by solid and side IDs."""
        for face in self.viewport.faces:
            if face.solid_id == solid_id and face.side_id == side_id:
                return face
        return None

    def generate_trigger(self, face, targetname):
        """Generates a trigger entity based on a selected face."""
        try:
            # Calculate normal
            normal = face.calculate_normal()
            if normal.magnitude() == 0:
                logging.warning(f"Face {face.side_id} has a degenerate normal. Skipping trigger generation.")
                return None

            # Extrude vertices
            extruded_vertices = [v + normal.scale(TRIGGER_EXTENSION_DISTANCE) for v in face.vertices]

            # Create sides
            sides = []
            num_vertices = len(face.vertices)
            for i in range(num_vertices):
                j = (i + 1) % num_vertices
                v1 = face.vertices[i]
                v2 = face.vertices[j]
                v3 = extruded_vertices[j]
                v4 = extruded_vertices[i]

                plane_str = f"({v1.x} {v1.y} {v1.z}) ({v2.x} {v2.y} {v2.z}) ({v3.x} {v3.y} {v3.z})"
                side_id = self.map_data.generate_unique_id()
                side = Side(
                    side_id=side_id,
                    plane_str=plane_str,
                    material="TOOLS/TOOLSTRIGGER",
                    vertices_plus=[v1, v2, v3, v4],
                    uaxis="[1 0 0 0] 0.25",
                    vaxis="[0 -1 0 0] 0.25",
                    rotation=0.0,
                    lightmapscale=16.0,
                    smoothing_groups=0
                )
                sides.append(side)

            # Front and Back sides
            front_plane_str = ' '.join([f"({v.x} {v.y} {v.z})" for v in face.vertices])
            back_plane_str = ' '.join([f"({v.x} {v.y} {v.z})" for v in reversed(extruded_vertices)])

            front_side_id = self.map_data.generate_unique_id()
            front_side = Side(
                side_id=front_side_id,
                plane_str=front_plane_str,
                material="TOOLS/TOOLSTRIGGER",
                vertices_plus=face.vertices,
                uaxis="[1 0 0 0] 0.25",
                vaxis="[0 -1 0 0] 0.25",
                rotation=0.0,
                lightmapscale=16.0,
                smoothing_groups=0
            )
            sides.append(front_side)

            back_side_id = self.map_data.generate_unique_id()
            back_side = Side(
                side_id=back_side_id,
                plane_str=back_plane_str,
                material="TOOLS/TOOLSTRIGGER",
                vertices_plus=extruded_vertices,
                uaxis="[1 0 0 0] 0.25",
                vaxis="[0 -1 0 0] 0.25",
                rotation=0.0,
                lightmapscale=16.0,
                smoothing_groups=0
            )
            sides.append(back_side)

            # Create solid
            trigger_solid_id = self.map_data.generate_unique_id()
            trigger_solid = Solid(solid_id=trigger_solid_id)
            trigger_solid.sides = sides
            trigger_solid.editor = {
                'color': '220 30 220',
                'visgroupshown': '1',
                'visgroupautoshown': '1',
                'logicalpos': '[0 0]'
            }

            # Create trigger entity
            centroid = np.mean([v.as_tuple() for v in face.vertices], axis=0)
            origin_x, origin_y, origin_z = centroid

            trigger_entity_id = self.map_data.generate_unique_id()
            trigger_entity = Entity(
                entity_id=trigger_entity_id,
                classname='trigger_multiple',
                spawnflags=4097,
                origin=Vertex(*centroid),
                properties={
                    'StartDisabled': '0',
                    'targetname': targetname,
                    'wait': '1'
                },
                connections={},
                editor={
                    'color': '220 30 220',
                    'visgroupshown': '1',
                    'visgroupautoshown': '1',
                    'logicalpos': '[0 1500]'
                },
                solids=[trigger_solid]
            )

            return trigger_entity
        except Exception as e:
            logging.error(f"Error generating trigger: {str(e)}")
            return None

    def execute_command(self, command):
        """Executes a command and updates undo/redo stacks."""
        command.execute()
        self.undo_stack.append(command)
        self.redo_stack.clear()
        self.update_undo_redo_buttons()
        self.viewport.update()

    def add_trigger(self):
        """Adds triggers to selected faces."""
        if not self.map_data:
            QMessageBox.warning(self, "Warning", "Please load a VMF file first.")
            return
        if not self.viewport.selected_faces:
            QMessageBox.warning(self, "Warning", "Please select at least one face.")
            return
        targetname = self.targetname_input.text()
        if not targetname:
            QMessageBox.warning(self, "Warning", "Please enter a targetname for the triggers.")
            return
        command = AddTriggerCommand(self, targetname, self.viewport.selected_faces)
        self.execute_command(command)

    def undo(self):
        """Undoes the last command."""
        if self.undo_stack:
            command = self.undo_stack.pop()
            command.undo()
            self.redo_stack.append(command)
            self.update_undo_redo_buttons()
            self.viewport.update()

    def redo(self):
        """Redoes the last undone command."""
        if self.redo_stack:
            command = self.redo_stack.pop()
            command.execute()
            self.undo_stack.append(command)
            self.update_undo_redo_buttons()
            self.viewport.update()

    def update_undo_redo_buttons(self):
        """Updates the state of undo/redo buttons."""
        self.undo_button.setEnabled(bool(self.undo_stack))
        self.redo_button.setEnabled(bool(self.redo_stack))

    def update_progress_bar(self, value):
        """Updates the progress bar."""
        self.progress_bar.setValue(value)
        QApplication.processEvents()  # Ensure the UI updates

    def focus_on_map(self):
        """Centers the camera on the loaded map."""
        if not self.solids:
            return
        all_vertices = [vertex.as_tuple() for solid in self.solids for side in solid.sides for vertex in side.vertices_plus]
        all_vertices = np.array(all_vertices)
        center = np.mean(all_vertices, axis=0)
        max_extent = np.max(np.linalg.norm(all_vertices - center, axis=1))
        self.viewport.camera.target = center
        self.viewport.camera.position = center + np.array([0.0, 0.0, max_extent * 2])
        self.viewport.update()

    def delete_selected_triggers(self):
        """Deletes the currently selected triggers."""
        if hasattr(self, 'selected_triggers') and self.selected_triggers:
            for trigger in self.selected_triggers:
                self.viewport.remove_trigger(trigger)
            self.update_trigger_list()
            self.selected_triggers = []
            self.update_info_label()
            self.viewport.update()

    def closeEvent(self, event):
        """Handles the window close event."""
        if self.map_data and self.viewport.triggers:
            reply = QMessageBox.question(self, 'Save Changes?',
                                         "Do you want to save your changes?",
                                         QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if reply == QMessageBox.Yes:
                self.save_vmf_file()
                event.accept()
            elif reply == QMessageBox.Cancel:
                event.ignore()
            else:
                event.accept()
        else:
            event.accept()

class ProgressEmitter(QObject):
    progressUpdated = pyqtSignal(int)

def main():
    """Main entry point of the application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
