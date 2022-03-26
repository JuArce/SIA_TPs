import sys

print('Argument List:', str(sys.argv))
assert len(sys.argv) == 2, 'Missing config json'


f = open(sys.argv[1], 'r')




f.close()
