import sys
sys.path.insert(1,'../')
sys.path.insert(1,'../../')
from firefly import *
from tqdm import trange, tqdm

# --------------------------------------------------------------------#
# make the gif
# --------------------------------------------------------------------#
col = random.sample(fun_colors,1)[0]
# Creates gif of two simulations side by side
# Takes in two arrays of color iterations
def lattice2D_mkgif_double(colors_it_one, colors_it_two, kappa, name="double_torus", freeze=False):
    col = random.sample(fun_colors,1)[0]
    v = n
    images = []
    def make_gif(frame_one, frame_two): 
        fig = plt.figure(figsize=(8,8))
        fig.add_subplot(1,2, 1)
        plt.pcolormesh(frame_one, cmap=col)
        plt.axis('square')

        plt.imshow(frame_one)

        fig.add_subplot(1,2, 2)
        plt.pcolormesh(frame_two, cmap=col)
        plt.axis('square')

        plt.imshow(frame_two)
        
        pngimg = io.BytesIO()
        plt.savefig(pngimg,format='png')
        plt.clf()
        pngimg.seek(0)
        img = Image.open(pngimg)
        images.append(img)

    # Starting generating frames
    len_one = len(colors_it_one)
    len_two = len(colors_it_two)
    if len_one > len_two:
        colors_it = colors_it_one
        colors_it_other = colors_it_two
    else:
        colors_it = colors_it_two
        colors_it_other = colors_it_one
    final_frame = []
    a = np.reshape(np.asarray(colors_it), (np.shape(colors_it)[0], n, n))
                # Add frame
    b = np.reshape(np.asarray(colors_it_other), (np.shape(colors_it_other)[0], n, n))
    for i, colors in enumerate(tqdm(colors_it)):
        if i % kappa == 0:
            if i <= len(colors_it_other):
                final_frame = b[i-1]
                make_gif(a[i-1],b[i-1])
            else:
                make_gif(a[i-1],final_frame)
    # append final frame
    make_gif(a[-1],b[-1])
    plt.close()
    tail = images[1:]; 
    # optionally freeze last frame
    if freeze:
        tail.extend([images[-1] for i in range(0,5)])
    images[0].save(name+'.gif', save_all=True, 
                    append_images=tail, optimize=False, duration=200, loop=0)

def main():
    n=31
    edgelist300 = edgeset_generator([n,n], show=False)
    kappa = 8; colorlist300 = np.random.randint(4, size=n^2)
    net300 = ColorNNetwork(colorlist300, edgelist300)
    a = simulate_FCA(net300, kappa, its=1600, tree=True, timesec=360)[0]
    lattice2D_mkgif(a, kappa, 
            interval=kappa, name="/mnt/l/home/31pow3_kappa8_everykappa_200frames", freeze=True, cap=1600)

if __name__ == "__main__":
    main()
