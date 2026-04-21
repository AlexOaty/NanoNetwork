import random

running_ratio = 0
curr_ratio = 0
total_flips = 0
total_heads = 0
total_counts = 0

for i in range(999999):
    total_flips += 1
    flip = random.randint(0,1)
    if(flip == 0):
        total_heads += 1

    curr_ratio = total_heads / total_flips

    if(curr_ratio > 0.5):
        running_ratio += curr_ratio
        total_flips = 0
        total_heads = 0
        curr_ratio = 0
        total_counts += 1


if total_counts == 0:
    total_counts = 1
ans = (running_ratio / total_counts) * 4
print(ans)
