from pathlib import Path
import xml.etree.ElementTree as ET

xml_path = Path("data/elevation/mola/megt00n090hb.xml")
root = ET.parse(xml_path).getroot()

print("\n=== COORDINATE METADATA ===")

for element in root.iter():
    tag = element.tag.split("}")[-1]
    if tag in [
        "west_bounding_coordinate",
        "east_bounding_coordinate",
        "north_bounding_coordinate",
        "south_bounding_coordinate",
        "pixel_resolution_x",
        "pixel_resolution_y",
        "minimum_latitude",
        "maximum_latitude",
    ]:
        print(tag, ":", element.text)