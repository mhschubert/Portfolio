import os
import copy
import re
import gc
from itertools import chain
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import imageio


def _set_figure_design(fig, ax, df, route, designpath='../data/HYP_50M_SR_W.tif'):
    ax.gridlines()
    # load fancy map image
    img = plt.imread(designpath)
    img_extent = (-180, 180, -90, 90)
    ax.imshow(img, extent=img_extent,
              origin='upper', transform=ccrs.PlateCarree(),
              interpolation='spline36', regrid_shape=400)

    dat_ordered = copy.deepcopy(df.iloc[route])
    longs = list(dat_ordered['CapitalLongitude'])
    longs = longs + [longs[0]]
    lats = list(dat_ordered['CapitalLatitude'])
    lats = lats + [lats[0]]
    nms = dat_ordered.CapitalName.to_list()
    ax.set_global()
    ax.add_feature(cartopy.feature.BORDERS, linestyle=':', alpha=1)
    return fig, ax, longs, lats, nms

def plot_globe_route(route:list, df:pd.DataFrame, globe_enum=-1, central_lat=20, central_long=0,
                     figdir='../figures', save=True, fancypath='../data/HYP_50M_SR_W.tif'):

    fig = plt.figure(figsize=(19.53/2, 18.55/2), num=1, clear=True)
    ax = plt.axes(projection= ccrs.Orthographic(central_latitude=central_lat,
                                                central_longitude=central_long))
    # make a transformer
    plat = ccrs.PlateCarree()
    # set figure stuff
    fig, ax, longs, lats, nms = _set_figure_design(fig, ax, df, route, fancypath)
    ax.plot(longs,
            lats,
            color='black', marker='o', markersize=3, linewidth=1,
            linestyle='--', transform=plat)
    # annotate every 10th capital
    for i, cap in enumerate(nms):
        if i%10 ==0:
            ax.text(longs[i], lats[i], s=cap, transform=plat, color='black')
    if globe_enum == -1:
        nm = 'view_lat_long_{}_{}'.format(central_lat, central_long)
    else:
        nm = 'part_{}'.format(globe_enum)
    if save:
        fig.savefig(os.path.join(figdir, 'globe_{}.png'.format(nm))
                    , dpi=150, transparent=True)
    if globe_enum == -1:
        plt.show()
    else:
        print('plotted globe view {}'.format(globe_enum))
        return fig, ax

def plot_globes(route:list, df:pd.DataFrame, ident:str, num_steps=200, dpi=150, figdir='../figures', fancypath='../data/HYP_50M_SR_W.tif'):

    lat_viw_range = [20.0,20.0]
    long_view_angle = [-180.0, 180.0]
    lats = np.linspace(lat_viw_range[0], lat_viw_range[1], num_steps)
    longs = np.linspace(long_view_angle[0], long_view_angle[1], num_steps)
    figdirt = os.path.join(figdir, 'gifparts/globes')
    os.makedirs(figdirt, exist_ok=True)
    for i in range(num_steps):
        fig, ax = plot_globe_route(route, df, globe_enum=i, central_lat=lats[i], central_long=longs[i], save=False, fancypath=fancypath)
        fig_size = fig.get_size_inches()
        bbox = matplotlib.transforms.Bbox([[0.0, 0.0], [fig_size[0], fig_size[1]]])
        fig.savefig(os.path.join(figdirt,'globe_gif_part_{}.png'.format(i))
                    , dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches=bbox, transparent=True)
        fig.clf()
        plt.close()
        gc.collect()

    create_gif(figdirt, 'globe_gif_part', figdir, 'globe_travel{}.gif'.format(ident))

def plot_map_lambert(route:list, df:pd.DataFrame, enum=-1, figdir='../figures', save=True, fancypath='../data/HYP_50M_SR_W.tif'):

    fig = plt.figure(figsize=(16, 8), num=1, clear=True)
    ax = plt.axes(projection= ccrs.LambertCylindrical())
    # make a transformer
    plat = ccrs.PlateCarree()
    # set figure stuff
    fig, ax, longs, lats, nms = _set_figure_design(fig, ax, df, route, fancypath)
    if enum == -1:
        ax.plot(longs,
                lats,
                color='black', marker='o', markersize=3, linewidth=1,
                linestyle='--', transform=plat)
        # annotate every 8th capital
        for i, cap in enumerate(nms):
            if i % 8 == 0:
                ax.text(longs[i], lats[i], s=cap, transform=plat, color='black')
    else:
        # plot all dots
        #ax.scatter(longs,
        #        lats,
        #        color='black', marker='o', transform=plat)
        if enum <3:
            #plot last three with lines
            ax.plot(longs[:(enum+1)],
                    lats[:(enum+1)],
                    linewidth=1,linestyle='--', transform=plat)
            for i, cap in enumerate(nms[:(enum+1)]):
                ax.text(longs[:(enum+1)][i], lats[:(enum+1)][i], s=cap, transform=plat, color='black')

        else:
            ax.plot(longs[(enum-2):(enum+1)],
                    lats[(enum-2):(enum+1)],
                    linewidth=1,linestyle='--', transform=plat)
            #annotate capitals
            for i, cap in enumerate(nms[(enum-2):(enum+1)]):
                ax.text(longs[(enum-2):(enum+1)][i], lats[(enum-2):(enum+1)][i], s=cap, transform=plat, color='black')

    if enum == -1:
        nm = 'lambert_total'
    else:
        nm = 'part_{}'.format(enum)
    if save:
        fig.savefig(os.path.join(figdir, '{}.png'.format(nm))
                    , dpi=150, transparent=True)
    if enum == -1:
        plt.show()
    else:
        print('plotted lambert view {}'.format(enum))
        return fig, ax

def plot_lamberts(route:list, df:pd.DataFrame, ident:str, dpi=80, figdir='../figures', fancypath='../data/HYP_50M_SR_W.tif'):
    figdirt = os.path.join(figdir, 'gifparts/lamberts')
    os.makedirs(figdirt, exist_ok=True)
    for i in range(len(route)):
        fig, ax = plot_map_lambert(route, df, i, figdir, save=False, fancypath=fancypath)
        fig_size = fig.get_size_inches()
        bbox = matplotlib.transforms.Bbox([[0.0, 0.0], [fig_size[0], fig_size[1]]])
        fig.savefig(os.path.join(figdirt, 'lamberts_{}.png'.format(i))
                    , dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches=bbox, transparent=True)
        fig.clf()
        plt.close()
        gc.collect()
    create_gif(figdirt, 'lamberts', figdir, 'lambert_travel{}.gif'.format(ident))


def create_gif(fig_dir:str, identifier:str, outdir, outname):

    # sort the .png files based on index used above
    images, image_file_names = [], []
    for file_name in os.listdir(fig_dir):
        if re.search(identifier,str(file_name)) and file_name.endswith('.png'):
            image_file_names.append(file_name)
    sorted_files = sorted(image_file_names, key=lambda y: int(y.split('_')[-1].split('.')[0]))

    # define some GIF parameters
    frame_length = 0.1  # seconds between frames
    end_pause = 0.1  # seconds to stay on last frame
    # loop through files, join them to image array, and write to GIF
    for ii in range(0, len(sorted_files)):
        file_path = os.path.join(fig_dir, sorted_files[ii])
        if ii == len(sorted_files) - 1:
            for jj in range(0, int(end_pause / frame_length)):
                images.append(imageio.imread(file_path))
        else:
            images.append(imageio.imread(file_path))
    # the duration is the time spent on each image (1/duration is frame rate)
    savename = os.path.join(outdir, outname)
    imageio.mimsave(savename, images, 'GIF', duration=frame_length)
    print('wrote gif...')

def plot_costs(costs:np.ndarray, figdir = '../figures', algo=''):
    plt.clf()
    fig, ax = plt.subplots(1, figsize=(8,8))
    ax.plot(range(1,len(costs)+1), costs/1000, color='blue', linestyle='-')
    ax.set_xlabel('Generations')
    ax.set_ylabel('Cost in km')
    ax.set_title('Cost Curve for {} algorithm'.format(algo))
    fig.savefig(os.path.join(figdir, 'cost_plot_{}.png'.format(algo)), dpi=150)