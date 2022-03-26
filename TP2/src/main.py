import array
import sys

print('Argument List:', str(sys.argv))
assert len(sys.argv) == 2, 'Missing config json'

max_weight: int
total_items: int

with open(sys.argv[1], 'r') as f:
    line = f.readline()
    aux: list[str] = line.split(',')
    total_items = int(aux[0])
    max_weight = int(aux[1])

    while line:
        line = f.readline()
        aux: list[str] = line.split(',')


        print(line)
    f.close()
