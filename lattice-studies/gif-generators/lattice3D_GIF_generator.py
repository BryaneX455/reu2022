import sys
from mpl_toolkits.mplot3d import axes3d
sys.path.insert(1,'../../')
sys.path.insert(1,'../')
from firefly import *
from tqdm import trange, tqdm
# --------------------------------------------------------------------#
# create the graph and run the simulations, collect coloring sequence
# --------------------------------------------------------------------#

# --------------------------------------------------------------------#
# make the gif
# --------------------------------------------------------------------#
def lattice3D_mkgif(colors_it, kappa, dim, path="/mnt/l/home/", name="Lattice3D",
        duration=200, blinkonly=False, interval=4, holdframes=0):
    images = []
    col = random.sample(fun_colors,1)[0]
    blink = floor((kappa-1)/2)

    def make_gif(a,b,c,d,angle): 
        ax = plt.gca(projection='3d')
        ax.set_facecolor("black")
        plt.axis('off')
        ax.scatter(a.ravel(),
                   b.ravel(),
                   c.ravel(),
                   c=d,
                   cmap=col,
                   s=10, alpha=0.2)
        ax.view_init(angle%360,(angle*2)%360)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.clf()
        buf.seek(0)
        im = Image.open(buf)
        images.append(im)

    # Starting generating frames
    a = []
    for i, colors in enumerate(tqdm(colors_it[0:len(colors_it)+kappa+1:interval])):
        volume = np.reshape(colors, (dim,dim,dim))
        xi = np.arange(volume.shape[0])[:, None, None]
        x = np.arange(volume.shape[0])[:, None, None]
        y = np.arange(volume.shape[1])[None, :, None]
        z = np.arange(volume.shape[2])[None, None, :]
        x, y, z = np.broadcast_arrays(x, y, z)
        c = np.tile(volume.ravel()[:, None], 1) 
 

        # Add frame
        make_gif(x,y,z,c,i)

    images.extend([images[-1]] * holdframes)

    # create the gif
    tail = images[1:]
    images[0].save(name+'.gif', save_all=True, 
                    append_images=tail, 
                    optimize=False, duration=duration, loop=0)

#anim = FuncAnimation(fig, animate, interval=500, frames=np.arange(0,len(a),kappa))
#anim.save('lattice2d.mp4')
n = 21
def main():

    edgelist = edgeset_generator([n,n,n], show=False)
    kappa=2; colorlist = np.random.randint(kappa, size=n**3)
    net= ColorNNetwork(colorlist, edgelist)
    a = simulate_FCA(net, kappa, its=2400, timesec=360000, tree=False)[0]

    with open('mysim.npy', 'wb') as f:
        np.save(f, np.array(a))
    with open('mysim.npy', 'rb') as f:
        b = np.load(f)

    lattice3D_mkgif(b, kappa, dim=n, name="test",
            duration=60, blinkonly=0, interval=kappa, holdframes=30)

if __name__ == "__main__":
    main()
