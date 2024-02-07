state_dict_1 = {'a': 1, 'b': 2, 'c': 3}
state_dict_2 = {'a': 4, 'b': 5, 'c': 6}

original_state_dict = {'a' : 4, 'b' : 2, 'c' : 0}
original_state_dict.update(state_dict_1)
print(original_state_dict)
original_state_dict.update(state_dict_2)
print(original_state_dict)