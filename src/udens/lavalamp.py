import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import numpy as np
from tqdm.notebook import tqdm as tqdm
import matplotlib.colors


def colormap(x):
    cmap = matplotlib.cm.get_cmap('viridis')(x)
    return [x, f'rgb{cmap[:-1]}']

norm = matplotlib.colors.Normalize()
colorscale = [colormap(i) for i in np.linspace(0, 1, 10)]

def lavalamp(post, obs, grid_coords, grid_low, grid_high):
    post = post.cpu().numpy()
    
    m_centers, m_edges, xy_centers, xy_edges = grid_coords
    X, Y, Z = torch.meshgrid(xy_centers, xy_centers, m_centers)
    values = np.transpose(post, [2, 1, 0])
    
    z_sub = obs['z_sub'].numpy()
    z_sub = z_sub[np.sum(np.abs(z_sub), axis = 1) != 0] 
    M_sub, x_sub, y_sub = z_sub.T
    
    im = np.array(obs['img'])
    im_x, im_y = im.shape
    x = np.linspace(grid_low[1], grid_high[1], im_x)
    y = np.linspace(grid_low[2], grid_high[2], im_y)
    z_low = grid_low[0]-1 if grid_low[0] == grid_high[0] else grid_low[0]
    z = np.ones(im.shape) * z_low
    

    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'is_3d': True}, {'is_3d': True}]],
                        subplot_titles=['Normal scale', 'Logarithmic scale'],
                        )

    for ncol, v, cbar_x in zip([1, 2], [values, np.log10(values)], [-0.10, None]):
        fig.add_trace(go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=v.flatten(),
            surface_count = 20,
        #     opacity = 0.1,
            opacityscale = [[0, 0], [1, 0.9]],
            colorbar_x=cbar_x,
        ), 1, ncol)

        fig.add_trace(go.Scatter3d(
            x = x_sub,
            y = y_sub,
            z = M_sub,
            mode ='markers',
            marker = dict(
                color = 'red',
                symbol = 'x',
                size = 5,
            ),
        ), 1, ncol)

    fig.add_trace(go.Surface(x=x, y=y, z=z,
        surfacecolor=im, 
        colorscale=colorscale,
        showscale=False,
    #     lighting_diffuse=1,
    #     lighting_ambient=1,
    #     lighting_fresnel=1,
    #     lighting_roughness=1,
    #     lighting_specular=0.5,
    ), 1, 1)


    fig.update_layout(
        height = 800, 
        width = 1600, 
        title_text="Subhalo posteriors",
        scene = dict(
            xaxis=dict(title=r"x"),
            yaxis=dict(title=r"y"),
            zaxis=dict(title=r'M'),
        ),
        showlegend=False
    )
    
    return fig