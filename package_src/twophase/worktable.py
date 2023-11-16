import twophase.coord as coord
import twophase.cubie as cubie

cube = cubie.CubieCube()
cube.randomize()
co, eo, ud_slice = cube.get_twist(), cube.get_flip(), cube.get_slice()
print(co, eo, ud_slice)
      
print(coord.CoordCube(cube).get_depth_phase1())