import subprocess
from subprocess import PIPE
from sys import stdout as out 
from sys import exit 
from time import sleep 
from os import getcwd

print('Building %s' % getcwd()) 
print('Pushing updates:')

x = subprocess.Popen([ 'rsync', '-ar', 
	'--exclude=obj', 
	'--exclude=bin',
	'.', 'sfm-build:~/build' ], stdout=PIPE)

out.write('| ')
while x.returncode is None:
	out.write('.')
	out.flush() 
	x.poll()
	sleep(0.5) 
print('\nDone.\n') 

if x.returncode != 0: 
	print('Failed with code %d' % x.returncode)
	exit(1) 


print('Making...')
out.flush() 

x = subprocess.Popen([ 'ssh', 'sfm-build', '/bin/bash' ], stdin=PIPE, stdout=PIPE)
x.stdin.write('''
cd build
make 1>&2 
''') 
x.stdin.close() 
x.wait() 

if x.returncode == 0:
	print('\n-- Success.')
else:
	print('Failed with status %d.' % x.returncode) 