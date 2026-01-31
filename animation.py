import matplotlib.pyplot as plt
import matplotlib.animation as anim
from PIL import Image
import numpy as np




def update_fig(*args):
        global i_fig
        if (i_fig<len(arr)-1):
            i_fig += 1
        else:
            i_fig=0
        im.set_array(arr[i_fig])
        #zi = z_span[i_fig]
        return im,

def update_fig2(*args):
        global i_fig
        if (i_fig<len(arr)-1):
            i_fig += 1
        else:
            i_fig=0
        im.set_array(arr[i_fig])
        im2.set_array(ang[i_fig])
        return im,im2


def animate_prop(space, dz, z_max, cmap ='hot', rescale_int=False, color_bar=False, plot=True, save_gif=None, save_npz=None, duration_frame=100):      #animate a propagation with step dz and maximum distance equal to z_max.
    global arr, im, i_fig, z_span                                             #Use rescale_int to set each frame such that the maximum intensity is equal to 1.
    arr = [np.abs(space.field)**2]
    if rescale_int == True:                                                   #descontinuar
        arr[0] = arr[0]/np.sqrt(np.amax(np.abs(arr[0])**2))
    z_span = np.linspace(0,z_max,int(z_max/dz))
    for i in z_span:
        space.propagate(dz)
        if rescale_int == True:
            space.field = space.field/np.sqrt(np.amax(np.abs(space.field)**2))
        arr.append(space.int_profile())
    data = np.array(arr)
    if save_npz != None:
        np.savez(save_gif, data)
    if save_gif !=None:
        data = data*255
        imgs = [Image.fromarray(img) for img in data]
        imgs[0].save(save_gif+'.gif', save_all=True, append_images=imgs[1:], duration=duration_frame, loop=0)
    if plot == True:
        fig = plt.figure()
        i_fig=0
        im = plt.imshow(arr[0], animated=True, cmap=cmap, extent=[-space.nix, space.nix, -space.niy, space.niy])
        ani = anim.FuncAnimation(fig, update_fig,  blit=True, cache_frame_data=False)
        if color_bar == True:
            plt.colorbar()
        plt.show()
    return space

def animate_prop2(space, dz, z_max, cmap ='hot', method='conv', rescale_int=False, color_bar=False, plot=True, save_gif=None, save_npz=None, duration_frame=100):      #animate a propagation with step dz and maximum distance equal to z_max.
    global arr, im, i_fig, z_span                                             #Use rescale_int to set each frame such that the maximum intensity is equal to 1.
    arr = [np.abs(space.field)**2]
    if rescale_int == True:
        arr[0] = arr[0]/np.sqrt(np.amax(np.abs(arr[0])**2))
    for i in range(int(z_max/dz)):
        space.propagate(dz, method=method)
        if rescale_int == True:
            space.field = space.field/np.sqrt(np.amax(np.abs(space.field)**2))
        arr.append(space.int_profile())
    data = np.array(arr)

    fig, ax = plt.subplots()
    i_fig=0
    im = plt.imshow(arr[0], animated=True, cmap=cmap)
    ani = anim.FuncAnimation(fig, update_fig,  blit=True, cache_frame_data=False, frames=100)
    if color_bar == True:
        plt.colorbar()
    if save_gif !=None:
        FFwriter = anim.PillowWriter(fps=20)
        ani.save(save_gif, writer = FFwriter)
    if save_npz != None:
        np.savez(save_npz, data)
    plt.show()
    return space

def animate_prop_phase(space, dz, z_max, cmap ='hot', cmap2 = 'gray', rescale_int=False, color_bar=False, plot=True, save_gif=None, save_npz=None, duration_frame=100):      
    global arr, ang, im, im2, i_fig, z_span                                               #animate a propagation and its phase with step dz and maximum distance equal to z_max.
    arr = [np.abs(space.field)**2]                                                                        
    ang = [np.angle(space.field)]
    if rescale_int == True:
        arr[0] = arr[0]/np.sqrt(np.amax(np.abs(arr[0])**2))
    z_span = np.linspace(0,z_max,int(z_max/dz))
    for i in z_span:
        space.propagate(dz)
        if rescale_int == True:
            space.field = space.field/np.sqrt(np.amax(np.abs(space.field)**2))
        arr.append(space.int_profile())
        ang.append(space.phase())
    data = np.array([arr, ang])
    if save_npz !=None:
        np.savez(save_gif, data)
    if save_gif !=None:
        data[0] = data[0]*255
        data[1] = (data[1] + np.pi)*255/(2*np.pi)
        data = np.concatenate((data[0], data[1]), axis=1)
        imgs = [Image.fromarray(img) for img in data]
        imgs[0].save(save_gif+'.gif', save_all=True, append_images=imgs[1:], duration=duration_frame, loop=0)
    if plot == True:
        fig, axarr = plt.subplots(1,2)
        i_fig=0
        im = axarr[0].imshow(arr[0], animated=True, cmap=cmap, extent=[-space.nix, space.nix, -space.niy, space.niy])
        im2 = axarr[1].imshow(ang[0], animated=True, cmap=cmap2, extent=[-space.nix, space.nix, -space.niy, space.niy])
        if color_bar == True:
            fig.colorbar(im)
            fig.colorbar(im2)
        ani = anim.FuncAnimation(fig, update_fig2,  blit=True, cache_frame_data=False)
        plt.show()
    return space