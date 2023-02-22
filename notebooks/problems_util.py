import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from shapely.geometry.point import Point
from shapely import affinity
from shapely.geometry import Polygon 


colors = ['red', 'blue']

def render_problem(problem, ellipses=None, title=None, fn=None, no_rewards=False):
    fig, ax = plt.subplots(dpi=144, figsize=(5, 5))
    X = []
    Y = []
    C = []
    for node_id, n in problem.get('nodes').items():
        X.append(n.get('X'))
        Y.append(n.get('Y'))   
        color = n.get('color') if not no_rewards else None
        if color is None:
            c = 'gray'
        else:
            c = colors[color]
        C.append(c)
    for link_key in problem.get('links').keys():
        src_id, tgt_id = link_key.split('_')
        src = problem.get('nodes')[src_id]
        tgt = problem.get('nodes')[tgt_id]        
        sx, sy = src.get('X'), src.get('Y')
        tx, ty = tgt.get('X'), tgt.get('Y')
        ax.plot([sx, tx], [sy, ty], c='gray', zorder=1)
    ax.scatter(X, Y, c=C, s=30, zorder=10)
    if ellipses:
        for color, ell in ellipses.items():
            ax.add_patch(Ellipse(ell['center'], ell['a'], ell['b'], angle=ell['angle'], ec=color, fill=False))
    
    if title:
        ax.set_title(title, fontsize=8)
    ax.set_axis_off()
    if fn:
        plt.savefig('./sample_maps/%s.png' % fn, facecolor='white')
    return ax
    

def ellipse_size(a, b):
    """
    Should always return a, since a is major axis
    """
    return max(a, b)
    

def create_ellipse(center, lengths, angle=0):
    """
    create a shapely ellipse. adapted from
    https://gis.stackexchange.com/a/243462
    """
    #     circ = Point(center).buffer(1)
    #     ell = affinity.scale(circ, lengths[0], lengths[1])
    #     ellr = affinity.rotate(ell, angle)
    
    # first draw the ellipse using matplotlib (interpolates, but decent approximation?)
    ellipse = Ellipse(center, lengths[0], lengths[1], angle) 
    vertices = ellipse.get_verts()     # get the vertices from the ellipse object
    ellipse = Polygon(vertices)        # Turn it into a polygon
    return ellipse

def ellipse_metrics(ellipses, keys=['red', 'blue']):
    geoms = []
    sizes = []
    for key in keys:
        ell = ellipses[key]
        geoms.append(create_ellipse(ell['center'], [ell['a'], ell['b']], angle=ell['angle']))
        sizes.append(ellipse_size(ell['a'], ell['b']))
    if geoms[0].disjoint(geoms[1]):
        overlap = 0
    else:
        intersect = geoms[0].intersection(geoms[1])
        overlap = intersect.area / (geoms[0].area + geoms[1].area)
    return sizes[0], sizes[1], overlap
