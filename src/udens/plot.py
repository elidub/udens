import torch, numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, NoNorm, Normalize
from mpl_toolkits.axes_grid1 import ImageGrid


def plt_imshow(plots, nrows = 1, x = 10, y = 8, 
               cmap = 'viridis', cbar = False, zlog = False,
               zlogmin = None, zlogmax = None,
               titles = None, title = None, title_size = 15,
               priors = None,
               msc_scatter = None, circle_scatter = None, circle_scatter2 = None,
               tl = False, tbl = None, tbl_title = None, fig_kwargs = {}, 
               vminmax = False, #vmin = None, vmax = None,
               setbkg = False,
               **imkwargs):
    """
    plots: list of what should be platted
    nrows: number of rows
    colobar: if True, a colorbar on each plot is plotted
    size_x, size_y: figsize = (size_x, size_y)
    """
    ncols = len(plots) // nrows
    
            
    if 'figsize' not in fig_kwargs: 
        fig, axes = plt.subplots(nrows = nrows, ncols = ncols, 
                                 figsize=(x, y), 
                                 **fig_kwargs)
    else: 
        fig, axes = plt.subplots(nrows = nrows, ncols = ncols, **fig_kwargs)
        
    fig.patch.set_facecolor('lightgray') if setbkg is True else fig.patch.set_facecolor('white')


    axes_flattened = [axes] if ncols == 1 else axes.flatten()
    
    if 'vmin' in imkwargs:
         vmin, vmax = imkwargs['vmin'], imkwargs['vmax']
    else:
        vmin, vmax = plots.min(), plots.max()
    
    
    if vminmax is not False:
        norm = LogNorm(vmin = vmin, vmax = vmax) if zlog is True else Normalize(vmin = vmin, vmax = vmax)
    else:
        imkwargs = imkwargs
        norm = LogNorm() if zlog is True else None
        
#     imkwargs = dict(**imkwargs, vmin = plots.min(), vmax = plots.max()) if vminmax is not False else imkwargs
    for i, (ax, plot) in enumerate(zip(axes_flattened, plots)):
        im = ax.imshow(plot, cmap = cmap, norm = norm, **imkwargs)

        
        if titles is not None: ax.set_title(titles[i], fontsize = title_size)    
        if title is not None:  fig.suptitle(title, fontsize = title_size, y = .95)#, x=.2, y = 1.05, fontweight = 'bold', horizontalalignment='left')
        if cbar is True:
            colorbar = fig.colorbar(im, fraction=0.046, pad=0.04, ax = ax)
            if zlog: colorbar.ax.set_yscale('log')
        
            if priors is not None:
                colorbar.ax.plot([0, 1], [priors[i]]*2, 'r', linewidth = 3)

    if circle_scatter2 is not None:
        r_max = np.max(circle_scatter2[:,0])
        radius = lambda r : np.exp(r-11.)
        
        for ss, ax in zip(circle_scatter2, axes_flattened):
            for s in ss:
                r = radius(s[0])
    #             print(s[0], r)

                ax.add_patch(plt.Circle((s[1], s[2]), r, color = 'red', alpha = 0.5))
        
        r_legends = radius(np.array([8., 9., 10.]))
        alphas = np.array([1.0, 0.5,.30])
        for r_legend, alpha in zip(r_legends, alphas):
            ax.add_patch(plt.Circle((2.45-r_legends[-1], -2.45+r_legends[-1]), r_legend, color = 'yellow', fill = False))
            
            
            
    if circle_scatter is not None:
        r_max = np.max(circle_scatter[:,0])
        radius = lambda r : np.exp(r-11.)
        
#         m_median = 10. #np.median(circle_scatter[:,0])
#         r_median = radius(m_median)
        
        for s in circle_scatter:
            r = radius(s[0])
#             print(s[0], r)
            
            ax.add_patch(plt.Circle((s[1], s[2]), r, color = 'red', alpha = 0.5))
        
        r_legends = radius(np.array([8., 9., 10.]))
        alphas = np.array([1.0, 0.5,.30])
        for r_legend, alpha in zip(r_legends, alphas):
            ax.add_patch(plt.Circle((2.45-r_legends[-1], -2.45+r_legends[-1]), r_legend, color = 'yellow', fill = False))
#         ax.text(2.5-r_median*2*1.5, -2.5+r_median*1.5, f'$M={m_median:.2f}$', 
#                 ha = 'right', va = 'center', fontsize = '14', color = 'white')
#         ax.text(2.5-r_median*1.5, -2.5+r_median*1.5, f'$M={m_median:.2f}$', 
#                 ha = 'center', va = 'center', fontsize = '14', color = 'white')


    if msc_scatter is not None:
        for s in msc_scatter:
            axes_flattened[int(s[0])].scatter(s[1], s[2], marker = 'x', s = 100, color = 'red', alpha = 0.5)
                
      
    if cbar == 'single':
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink = 0.8)
    if tl: plt.tight_layout() # tl stands for Tight Layout 
        
        
    if tbl is not None:
#         pass
        return tbl.experiment.add_figure(tbl_title, fig)
#     elif tbl is False:
#          return fig   
    else:
        plt.show()
        return fig  
    
    
    
    
    
    
    
    
    
    
    
    
# def plt_imshow_single(plots, nrows = 1, x = 10, y = 8, 
#                cmap = None, cbar = False, zlog = False,
#                zlogmin = None, zlogmax = None,
#                titles = None, title = None, title_size = 15,
#                priors = None,
#                msc_scatter = None, circle_scatter = None, circle_scatter2 = None,
#                tl = True, tbl = None, tbl_title = None, fig_kwargs = {}, vminmax = False, **imkwargs):
#     """
#     plots: list of what should be platted
#     nrows: number of rows
#     colobar: if True, a colorbar on each plot is plotted
#     size_x, size_y: figsize = (size_x, size_y)
#     """
    
#     ncols = len(plots) // nrows
    
#     fig_kwargs = dict(**fig_kwargs, figsize=(x, y)) if 'figsize' not in fig_kwargs else fig_kwargs
            
#     fig, axes = plt.subplots(nrows = nrows, ncols = ncols, **fig_kwargs)
        
#     fig.patch.set_facecolor('white')

#     if tl: plt.tight_layout() # tl stands for Tight Layout 

#     grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
#                  nrows_ncols=(nrows, ncols),
#                  axes_pad=0.15,
#                  share_all=True,
#                  cbar_location="right",
#                  cbar_mode="single",
#                  cbar_size="7%",
#                  cbar_pad=0.15,
#                  )
    
#     vmin, vmax = plots.min(), plots.max()
    
#     norm = LogNorm(vmin = vmin, vmax = vmax) if zlog is True else None
    
#     for i, (ax, plot) in enumerate(zip(grid, plots)):
#         im = ax.imshow(plot, norm = norm, **imkwargs)

        
#         if titles is not None: ax.set_title(titles[i], fontsize = title_size)    
#         if title is not None:  fig.suptitle(title, fontsize = title_size)
# #             colorbar = fig.colorbar(im, fraction=0.046, pad=0.04, ax = ax)
# #             if zlog: colorbar.ax.set_yscale('log')
        
# #         if priors is not None:
# #             colorbar.ax.plot([0, 1], [priors[i]]*2, 'r', linewidth = 3)

    

#     if circle_scatter2 is not None:
#         r_max = np.max(circle_scatter2[:,0])
#         radius = lambda r : np.exp(r-11.)
        
#         for ss, ax in zip(circle_scatter2, axes_flattened):
#             for s in ss:
#                 r = radius(s[0])
#                 ax.add_patch(plt.Circle((s[1], s[2]), r, color = 'red', alpha = 0.5))
        
#         r_legends = radius(np.array([8., 9., 10.]))
#         alphas = np.array([1.0, 0.5,.30])
#         for r_legend, alpha in zip(r_legends, alphas):
#             ax.add_patch(plt.Circle((2.45-r_legends[-1], -2.45+r_legends[-1]), r_legend, color = 'yellow', fill = False))
            
#     if circle_scatter is not None:
#         assert N == 1
#         r_max = np.max(circle_scatter[:,0])
#         radius = lambda r : np.exp(r-11.)
#         for s in circle_scatter:
#             r = radius(s[0])
#             ax.add_patch(plt.Circle((s[1], s[2]), r, color = 'red', alpha = 0.5))
        
#         r_legends = radius(np.array([8., 9., 10.]))
#         alphas = np.array([1.0, 0.5,.30])
#         for r_legend, alpha in zip(r_legends, alphas):
#             ax.add_patch(plt.Circle((2.45-r_legends[-1], -2.45+r_legends[-1]), r_legend, color = 'yellow', fill = False))

#     if msc_scatter is not None:
#         for s in msc_scatter:
#             axes_flattened[int(s[0])].scatter(s[1], s[2], marker = 'x', s = 100, color = 'red')
                
      
        
        
#     if tbl is not None:
#         return tbl.experiment.add_figure(tbl_title, fig)
#     else:
#         plt.show()
#         return fig  