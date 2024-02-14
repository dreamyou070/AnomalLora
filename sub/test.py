img_list = [1,2,3,1,2,3,1,2,3]
aug = [3,4,5]
for i in range(len(img_list)):
    aug_index = i % len(aug)
    print(aug_index)
