import sys
sys.path.insert(1,'../')
from firefly import * 
from tqdm import trange, tqdm
from matplotlib import gridspec 

# --------------------------------------------------------------------#
# make the gif
# --------------------------------------------------------------------#
def lattice4D_mkgif(colors_it, kappa, dim, slices, name="Lattice4D",
        duration=200, blinkonly=False, interval=4, holdframes=0):
    images = []
    col = random.sample(fun_colors,1)[0]
    blink = floor((kappa-1)/2)

    fig = plt.figure(figsize=(len(slices)*2,8))
    gs = gridspec.GridSpec(nrows=4, ncols=len(slices))

    def abcd(volume, index, sub):
        if sub // len(slices) == 0:
            data = volume[index,:,:,:]
        elif sub // len(slices) == 1:
            data = volume[:,index,:,:]
        elif sub // len(slices) == 2:
            data = volume[:,:,index,:]
        else:
            data = volume[:,:,:,index]
        a = np.arange(data.shape[0])[:, None, None]
        b = np.arange(data.shape[1])[None, :, None]
        c = np.arange(data.shape[2])[None, None, :]
        a, b, c = np.broadcast_arrays(a, b, c)
        d = np.tile(data.ravel()[:, None], 1) 

        return a,b,c,d

    def subplotslice(volume, index, sub):

        a,b,c,d = abcd(volume, index, sub)
        ax1 = fig.add_subplot(gs[sub],projection='3d')
        ax1.set_facecolor("black")
        ax1.scatter(a.ravel(),
                   b.ravel(),
                   c.ravel(),
                   c=d, cmap=col,
                   s=10, alpha=0.10)
        plt.axis('off')

    def make_gif(volume): 

        for sub, index in enumerate(slices*4):
            subplotslice(volume, index, sub)

        plt.subplots_adjust(wspace=0,hspace=0)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.clf()
        buf.seek(0)
        im = Image.open(buf)
        images.append(im)

    # Starting generating frames
    a = []
    for i, colors in enumerate(tqdm(colors_it[0:len(colors_it):interval])):
        volume = np.reshape(colors, (dim,dim,dim,dim))
        make_gif(volume)
    plt.close()

    images.extend([images[-1]] * holdframes)

    # create the gif
    tail = images[1:]
    images[0].save(name+'.gif', save_all=True, 
                    append_images=tail, optimize=False, duration=duration, loop=0)

#anim = FuncAnimation(fig, animate, interval=500, frames=np.arange(0,len(a),kappa))
#anim.save('lattice2d.mp4')
n = 15
edgelist300 = edgeset_generator([n,n,n,n], show=False)
kappa = 8; colorlist300 = np.random.randint(kappa, size=n**4)
net300 = ColorNNetwork(colorlist300, edgelist300)
a = simulate_FCA(net300, kappa, its=1000, timesec=360000, tree=False)[0]

lattice4D_mkgif(a, kappa, dim=n,
        slices=[0,2,4,6,8,10,12,14],name='kappa8_everykappa_10pow4',#,12,14], name="k4_13pow4_everyk",
        duration=175, blinkonly=0, interval=4, holdframes=10)
