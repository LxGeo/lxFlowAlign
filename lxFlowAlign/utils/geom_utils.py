
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
import random


def perturbate_vertcies(geometry, perturbation_std=0.5):
    """
    Applies spatial perturbation of geometry vertcies following rule defined.
    Input: Any shapely geometry
    Output: Any shapely geometry
    """

    random_perturbation = lambda xy : (xy[0]+random.normalvariate(0,perturbation_std), xy[1]+random.normalvariate(0,perturbation_std))

    geom_type = geometry.geom_type

    if geom_type == "Point":
        return Point(
            random_perturbation((geom_type.x, geom_type.y))
        )
    
    if geom_type == "LineString":
        return LineString(
            list(map(random_perturbation, geometry.coords))
        )

    if geom_type == "Polygon":
        return Polygon(
            list(map(random_perturbation, geometry.exterior.coords)),
            [list(map(random_perturbation, inner.coords)) for inner in geometry.interiors]
        )
    
    if geom_type == "MultiPolygon":
        return MultiPolygon([perturbate_vertcies(p) for p in geometry])
    
    raise "Unknown geometry!"