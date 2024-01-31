import twophase.coord as coord
import twophase.cubie as cubie

move_dict = {1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0}


for i in range(18):
    cube = cubie.CubieCube()
    cube.multiply(cubie.moveCube[i])
    coord_cube = coord.CoordCube(cube)
    # get_slice_sorted() => 0 <= slice_sorted < 11880 in phase 1
    co, eo, ud = cube.get_twist(), cube.get_flip(), cube.get_slice_sorted()
    if coord_cube.get_depth_phase1() == 1:
        move_dict[1] += 1
        
        
for i in range(18):
    for j in range(18):
        cube = cubie.CubieCube()
        cube.multiply(cubie.moveCube[i])
        cube.multiply(cubie.moveCube[j])
        coord_cube = coord.CoordCube(cube)
        # get_slice_sorted() => 0 <= slice_sorted < 11880 in phase 1
        co, eo, ud = cube.get_twist(), cube.get_flip(), cube.get_slice_sorted()
        if coord_cube.get_depth_phase1() == 2:
            move_dict[2] += 1
            
for i in range(18):
    for j in range(18):
        for k in range(18):
            cube = cubie.CubieCube()
            cube.multiply(cubie.moveCube[i])
            cube.multiply(cubie.moveCube[j])
            cube.multiply(cubie.moveCube[k])
            coord_cube = coord.CoordCube(cube)
            # get_slice_sorted() => 0 <= slice_sorted < 11880 in phase 1
            co, eo, ud = cube.get_twist(), cube.get_flip(), cube.get_slice_sorted()
            if coord_cube.get_depth_phase1() == 3:
                move_dict[3] += 1
                
for i in range(18):
    for j in range(18):
        for k in range(18):
            for l in range(18):
                cube = cubie.CubieCube()
                cube.multiply(cubie.moveCube[i])
                cube.multiply(cubie.moveCube[j])
                cube.multiply(cubie.moveCube[k])
                cube.multiply(cubie.moveCube[l])
                coord_cube = coord.CoordCube(cube)
                # get_slice_sorted() => 0 <= slice_sorted < 11880 in phase 1
                co, eo, ud = cube.get_twist(), cube.get_flip(), cube.get_slice_sorted()
                if coord_cube.get_depth_phase1() == 4:
                    move_dict[4] += 1
                    
for i in range(18):
    for j in range(18):
        for k in range(18):
            for l in range(18):
                for m in range(18):
                    cube = cubie.CubieCube()
                    cube.multiply(cubie.moveCube[i])
                    cube.multiply(cubie.moveCube[j])
                    cube.multiply(cubie.moveCube[k])
                    cube.multiply(cubie.moveCube[l])
                    cube.multiply(cubie.moveCube[m])
                    coord_cube = coord.CoordCube(cube)
                    # get_slice_sorted() => 0 <= slice_sorted < 11880 in phase 1
                    co, eo, ud = cube.get_twist(), cube.get_flip(), cube.get_slice_sorted()
                    if coord_cube.get_depth_phase1() == 5:
                        move_dict[5] += 1
                        
print(move_dict)