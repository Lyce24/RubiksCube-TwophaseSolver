import twophase.coord as coord
import twophase.cubie as cubie

move_dict = {}

for i in range(10000):
    cube = cubie.CubieCube()
    cube.randomize()

    if coord.CoordCube(cube).get_depth_phase1() not in move_dict:
        move_dict[coord.CoordCube(cube).get_depth_phase1()] = 1
    else:
        move_dict[coord.CoordCube(cube).get_depth_phase1()] += 1
        
print(move_dict)

