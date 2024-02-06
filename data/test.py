import numpy as np
import random
dir_list = [1,2,3]
repeat = 3
a_list = [ a for a in dir_list for i in range(repeat)]
print(a_list)
random.shuffle(a_list)
print(a_list)