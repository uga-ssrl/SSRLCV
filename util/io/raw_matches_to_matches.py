# this is the bash script to run
# ./bin/sift_cli images/everest01.png > p01.kp; ./bin/sift_cli images/everest02.png > p02.kp; ./bin/match_cli p01.kp p02.kp > matches_raw.txt

raw_matches = []

with open('matches_raw.txt') as f:
    for line in f:
        items = line.split( )
        raw_matches.append([items[0],items[1],items[4],items[5]])

print raw_matches

match_file = open('matches.txt', 'w')
match_file.write("%s\n" % str(len(raw_matches)-1))

for item in raw_matches:
    write_string = "0001.jpg,0002.jpg," + str(item[0]) + "," + str(item[1]) + "," + str(item[2]) + "," + str(item[3]) + ",0,255,0\n"
    match_file.write(write_string)
