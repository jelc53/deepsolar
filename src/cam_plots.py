import pickle
#import matplotlib.pyplot as plt

def read_from_pickle(path):
    output = None
    with open(path, 'rb') as file:
        try:
            while True:
                output =  pickle.load(file)
        except EOFError:
            pass
    return output

def find_non_zero_cam(cams):
    non_zero_cams = []
    for cam in cams:
        if cam[0].sum() > 0:
            non_zero_cams.append(cam)
    return non_zero_cams

if __name__ == '__main':
    out = read_from_pickle('CAM_list.pickle')
    nonzero_out = find_non_zero_cam(out)

    for nz in nonzero_out:
        print(nz[1])

    #plt.imshow(out[-1][0])
    #plt.show()
