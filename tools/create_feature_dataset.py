
import random

# 2. through hole 
def create_through_hole():
    diameter = random.randrange(10, 30)
    x = random.randrange(-30, 30)
    y = random.randrange(-30, 30)
    # make the base
    result = cq.Workplane("XY").box(100, 100, 100)\
        .faces(">Z").workplane().center(x,y).hole(diameter)
    show_object(result)
    return result

# 3. blind hole
def create_blind_hole():
    diameter = random.randrange(10, 30)
    depth = random.randrange(10, 30)
    x = random.randrange(-30, 30)
    y = random.randrange(-30, 30)
    result = cq.Workplane("XY").box(100, 100, 100)\
        .faces(">Z").workplane().center(x,y).hole(diameter, 50)
    show_object(result)
    return result


# 5. rectangular passage
def create_rectangular_passage():
    x = random.randrange(-30, 30)
    y = random.randrange(-30, 30)
    width = random.randrange(10, 30)
    height = random.randrange(10, 30)
    result = cq.Workplane("XY").box(100, 100, 100)\
        .faces(">Z").workplane().center(x,y).rect(width, height).cutThruAll()
    show_object(result)
    return result

# 10. Triangular pocket
def create_triangular_pocket():
    diameter = random.randrange(20, 40)
    x = random.randrange(-30, 30)
    y = random.randrange(-30, 30)
    result = cq.Workplane("XY").box(100, 100, 100).center(x, y).polygon(3, diameter).cutThruAll()
    show_object(result)
    return result

# 20. round()
def create_round():
    diameter = random.randrange(10, 70)
    result = cq.Workplane("XY").box(100, 100, 100).edges(">Zand>X").fillet(diameter)
    show_object(result)
    return result

# 24. 6-sides pocket
def create_6sides_pocket(): 
    diameter = random.randrange(20, 40)
    x = random.randrange(-30, 30)
    y = random.randrange(-30, 30)
    result = cq.Workplane("XY").box(100, 100, 100).center(x, y).polygon(6, diameter).cutThruAll()
    show_object(result)
    return result

OUTPUT_PATH = "/home/seung/Workspace/project/samsung/Ejector-Pin-Estimation-SAMSUNG/datasets/cad_sample/"


for i in range(500):

    result = create_through_hole()
    cq.exporters.export(result, OUTPUT_PATH + 'through_hole/{}.step'.format(i))

    result = create_blind_hole()
    cq.exporters.export(result, OUTPUT_PATH + 'blind_hole/{}.step'.format(i))

    result = create_rectangular_passage()
    cq.exporters.export(result, OUTPUT_PATH + 'rectangular_passage/{}.step'.format(i))

    result = create_triangular_pocket()
    cq.exporters.export(result, OUTPUT_PATH + 'triangular_pocket/{}.step'.format(i))

    result = create_round()
    cq.exporters.export(result, OUTPUT_PATH + 'round/{}.step'.format(i))

    result = create_6sides_pocket()
    cq.exporters.export(result, OUTPUT_PATH + '6sides_pocket/{}.step'.format(i))







