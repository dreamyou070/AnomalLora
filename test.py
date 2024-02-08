import random

#sorted(glob.glob(anomaly_source_path+f"/*/*.{ext}"))
b = []
a = [1,3,4]
c = [2,5,6]
d = sorted(a)
d2 = sorted(c)
b.extend(d)
b.extend(d2)
print(f'b: {b}')