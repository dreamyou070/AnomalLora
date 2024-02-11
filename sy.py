p = 0

total_step = 100
thred = p

if p == 0:
    p = p + 0.00001
devide_num = 1/(p)
print(devide_num)
for step in range(total_step):
    if step % devide_num == 0:
        print(step)
    else:
        continue