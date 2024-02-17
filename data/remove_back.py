
from PIL import Image
import numpy as np
from rembg import remove
def remove_bg(input_path, output_path):
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input)
            o.write(output)

def main():

    input_path = '000.png'
    output_path = '000_rm.png'
    remove_bg(input_path, output_path)
    output_img = np.array(Image.open(output_path))
    h, w = output_img.shape[0], output_img.shape[1]
    alpha_channel = output_img[:,:,3]
    mask_np = np.zeros((h,w)) # 투명한 곳은 0, 아닌 곳은 1
    for h_index in range(h) :
        for w_index in range(w) :
            alpha_value = alpha_channel[h_index,w_index]
            if alpha_value == 0 :
                mask_np[h_index,w_index] = 0
            else :
                mask_np[h_index,w_index] = 255
    mask_pil = Image.fromarray(mask_np.astype(np.uint8))
    mask_pil.save('000_mask.png')




if __name__ == "__main__":
    main()
