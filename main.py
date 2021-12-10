# -*- coding: utf-8 -*-
import os
import pickle
import pandas as pd
import numpy as np
from IPython import display, get_ipython
from numpy.random import randint, default_rng, choice

# For ploting
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

# For  procesing
from numba import njit

############### Function definitions ###############

def data_curve():
    # Read data from a csv
    data = pd.read_csv('DatosOrdenados.csv', header=0, index_col=0)
    # Order data
    z = np.array(data.values)
    # Get ranges
    x_range = np.arange(z.shape[0])
    y_range = np.arange(z.shape[1])
    # Get grid type values
    X, Y = np.meshgrid(x_range, y_range)
    #Z = z[X,Y]
    # Get labels
    x_labels, y_labels = data.axes
    return (x_range, y_range, x_labels, y_labels, X, Y, z)

# Function to retrieve the generated curve
@njit(parallel=True)
def get_curve(x_range, y_range, c):
    points = np.zeros((len(x_range), len(y_range)))
    
    ##################################################
    mx = []
    my = []
    dex = []
    dey = []
    p = []
    q = []
    r = []
    for gen in c:
        if len(mx) < x_curves:
            mx.append(gen)
        elif len(my) < y_curves:
            my.append(gen)
        elif len(dex) < x_curves:
            dex.append(gen or 0.01)
        elif len(dey) < y_curves:
            dey.append(gen or 0.01)
        elif len(p) < x_curves * y_curves:
            p.append(gen)
        elif len(q) < x_curves * y_curves:
            q.append(gen)
        elif len(r) < x_curves * y_curves:
            r.append(gen)
    
    for x in x_range:
        x_temp = x / x_range[-1];
        for y in y_range:
            y_temp = y / y_range[-1];
            
            mfx = []
            mfy = []
            for i in range(x_curves):
                mfx.append(np.exp(( -(x_temp - mx[i])**2) / (2 * dex[i]**2)))
            for i in range(y_curves):
                mfy.append(np.exp(( -(y_temp - my[i])**2) / (2 * dey[i]**2)))
            
            inf = []
            reg = []
            for i in range(x_curves):
                for j in range(y_curves):
                    inf_val = mfx[i] * mfy[j]
                    inf.append(inf_val)
                    item = (i * y_curves) + j
                    reg.append(inf_val * (p[item] * x_temp + q[item] * y_temp + r[item]))

            a = 0
            b = 0
            for value in reg:
              a += value
            for value in inf:
              b += value
            
            points[x, y] = (a / b)

    return points

def get_mesh_curve(x_range, y_range, c):
    X, Y = np.meshgrid(x_range, y_range)
    points = get_curve(x_range, y_range, c)
    return points[X, Y]

def get_mf(x_range, y_range, c):
    ##################################################
    mx = []
    my = []
    dex = []
    dey = []
    for gen in c:
        if len(mx) < x_curves:
            mx.append(gen)
        elif len(my) < y_curves:
            my.append(gen)
        elif len(dex) < x_curves:
            dex.append(gen or 0.01)
        elif len(dey) < y_curves:
            dey.append(gen or 0.01)

    mfx = [ [] for _ in range(x_curves) ]
    mfy = [ [] for _ in range(y_curves) ]
    
    for x in x_range:
        x_temp = x / x_range[-1];
        for i in range(x_curves):
            mfx[i].append(np.exp(( -(x_temp - mx[i])**2) / (2 * dex[i]**2)))
            
    for y in y_range:
        y_temp = y / y_range[-1];
        for i in range(y_curves):
            mfy[i].append(np.exp(( -(y_temp - my[i])**2) / (2 * dey[i]**2)))
    
    return (mfx, mfy)

# Returns the chromsomes values decoded by w
@njit
def decoded_values(chromosome_values, w):
    normalized = 2 * y_curves + 2 * x_curves
    return np.array([x / 255 for x in chromosome_values[:normalized]]
        + [x / w for x in chromosome_values[normalized:]])

# Sets the aptitude value of a chromosome (Area between the curve and the original curve)
@njit
def calculate_aptitude(x_range, y_range, chromosome_values, original_curve, w):
    decoded = decoded_values(chromosome_values, w)
    aprox_curve = get_curve(x_range, y_range, decoded)
    return np.abs(original_curve - aprox_curve).sum()
    
# Returns the best chromosome from the competitors (Minimum sum of distances)
def get_best_chromosome(population, competitors_ids):
    return min(competitors_ids, key=lambda chromosome_id: population[chromosome_id]['aptitude'])

# Generates a random filled chromosome 
def random_chromosome(max_value, num_of_genes):
    return {'values': np.array(default_rng().choice(max_value + 1, size=num_of_genes, replace = True).astype(int)),
            'aptitude': 0}

# Reproduce two chromosomes
def reproduce(chromosome_father, chromosome_mother):
    # Spliting by 8 bits per value
    father = chromosome_father['values'].tolist()
    mother = chromosome_mother['values'].tolist()
    num_genes = len(father)
    partition = randint(1, 8 * num_genes - 1)
    mod_partition = partition % 8
    position = partition // 8
    
    if (mod_partition != 0):
        f = father[position]
        m = mother[position]
        
        low_mask = (2 ** mod_partition) - 1
        high_mask = 255 - low_mask
        
        x = f & high_mask
        y = f & low_mask
        xx = m & high_mask
        yy = m & low_mask
        
        xyy = x | yy
        xxy = xx | y
    
    x = father[:position]
    y = father[position:]
    xx = mother[:position]
    yy = mother[position:]
    
    child1 = x + yy
    child2 = xx + y
    
    if (mod_partition != 0):
        child1[position] = xyy
        child2[position] = xxy
        
    return [
        {'values': np.array(child1), 'aptitude': 0}, 
        {'values': np.array(child2), 'aptitude': 0}]

# Mutates a randum number of genes in a chromosome
def mutate(chromosome):
    values = chromosome['values']
    num_genes = len(chromosome['values'])
    num_mutations = randint(num_genes) + 1
    
    for i in range(num_mutations):

        partition = randint(0, 8 * num_genes)
        mod_partition = partition % 8
        position = partition // 8
        
        if values[position] == 0:
            values[position] += 1
        elif values[position] == 255:
            values[position] -= 1
        elif (choice([True, False])):
            mask = 1 << mod_partition
            values[position] = values[position] ^ mask
        elif (choice([True, False])):
            values[position] += choice([1, -1])
        else:
            values[position] = randint(255 + 1)
            
    chromosome['values'] = values

def print_plots(x_range, y_range, original_curve, aptitudes, c_values, use_plotly = False, use_inline = True):
    
    X, Y = np.meshgrid(x_range, y_range)
    
    graph_definition = 200
    x_range_new = np.arange(x_range[-1], step=x_range[-1]/graph_definition)
    y_range_new = np.arange(y_range[-1], step=y_range[-1]/graph_definition)
    
    aprox_curve = get_mesh_curve(x_range, y_range, c_values)
    
    if use_plotly:
        
        # Initialize figure with 4 subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'xy'}],
                   [{'type': 'xy'}, {'type': 'xy'}]])
        
        fig.add_trace(
            go.Surface(z=aprox_curve, x=X, y=Y, colorscale='viridis', opacity=0.75),
            row=1, col=1)
        
        fig.add_trace(
            go.Surface(z=original_curve, x=X, y=Y),
            row=1, col=1)
        
        fig.add_trace(go.Scatter(y=aptitudes), row=1, col=2)
        
        (mfx, mfy) = get_mf(x_range_new, y_range_new, c_values)
        for mf in mfx:
            fig.add_trace(go.Scatter(y=mf, x=x_range_new), row=2, col=1)
        for mf in mfy:
            fig.add_trace(go.Scatter(y=mf, x=y_range_new), row=2, col=2)

        fig.update_layout(title = 'Ocupation',
                          autosize = True,
                          margin = dict(l=65, r=50, b=65, t=90),
                          scene = dict(
                              xaxis = dict(
                                  showticklabels = False,
                                  ticktext = x_labels,
                                  tickvals = x_range,
                                  title = dict(
                                      text = 'Fecha')
                                  ),
                              yaxis = dict(
                                  showaxeslabels = False,
                                  ticktext = y_labels,
                                  tickvals = y_range,
                                  title = dict(
                                      text = '')
                                  ),
                              zaxis = dict(
                                  ticks='outside',
                                  ticksuffix='%',
                                  title = dict(
                                      text = 'Population')
                                  ),
                              ),
                          )
        
        if use_inline:
            display.clear_output(wait = True)
            fig.show()
        else:
            fig.show(renderer="browser")
    else:
        if use_inline:
            matplotlib.use('module://matplotlib_inline.backend_inline')
        else:
            matplotlib.use('Qt5Agg')
        
        fig = plt.figure(figsize = (18, 8))
        
        # -------------- 3D surface
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.set_xlabel('Fecha')
        ax1.set_xticklabels(x_labels)
        ax1.set_ylabel('')
        ax1.set_yticks(y_range)
        ax1.set_yticklabels(y_labels)
        ax1.set_zlabel('Population %')
        
        #TODO: Add a color bar which maps values to colors.
        #fig2.colorbar(surf, shrink=0.5, aspect=5)
    
        # Plot the surface.
        ax1.plot_surface(X, Y, original_curve, cmap=plt.cm.plasma, alpha=0.5)
        ax1.plot_surface(X, Y, aprox_curve, cmap=plt.cm.viridis, alpha=0.5)
        
        # -------------- Aptitude curve
        ax2 = fig.add_subplot(2,2,2)
        ax2.clear()
        ax2.plot(aptitudes, linewidth=2, color='royalblue', label='Aptitude of generation {}'.format(len(aptitudes)), path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()])
        ax2.grid(linestyle='--')
        ax2.legend()
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Aptitude')
        ax2.title.set_text("Aptitude: {:.2f}".format(aptitudes[-1]))
        
        # -------------- Mf curves
        (mfx, mfy) = get_mf(x_range_new, y_range_new, c_values)
        ax3 = fig.add_subplot(2,2,3)
        ax3.clear()
        for item in range(len(mfx)):
            ax3.plot(x_range_new, mfx[item], linewidth=2, label='mfx{}'.format(item), path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()])
        ax3.set_xlabel('Fecha (x)')
        ax3.set_ylabel('Population (z)')
        ax3.legend()
        
        ax4 = fig.add_subplot(2,2,4)
        ax4.clear()
        for item in range(len(mfy)):
            ax4.plot(y_range_new, mfy[item], linewidth=2, label='mfy{}'.format(item), path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()])
        ax4.set_xlabel('Clases (y)')
        ax4.set_ylabel('Population (z)')
        ax4.legend()

        plt.show()

############################# Read Original Data ##############################

(x_range, y_range, x_labels, y_labels, X, Y, original_curve) = data_curve()

original_mesh_curve = original_curve[X,Y]

# ----------------------------- Variable Initialization ----------------------------- 
num_of_chromosomes = 100            # Equals to the number of chromosomes per population
num_of_generations = 800            # Number of generations to reproduce
                                    # Equals to the number of genes
x_curves = 3
y_curves = 3
num_of_genes = 3 * x_curves * y_curves + 2 * y_curves + 2 * x_curves
percentage_of_opponents = 0.05      # Percentage of chromosomes from the population to compite
weight = 255 / (original_curve.max() * 1.1)      # To set a max value of our data
tries = 10

# Mutation/Elitism variables
allow_mutation = True
mutation_percentage = 0.3
allow_elitism = True

# History variables
aptitudes = []
best_chromosomes = []
previous_generation = []
load_previous = True
save_state = True
check_point = 10
    
# Load history state
history_file = 'history_{}_{}_{}.pth'.format(x_curves, y_curves, num_of_chromosomes)
if load_previous and os.path.exists(history_file):
    with open(history_file, 'rb') as fp:   # Unpickling
        print('Loaded previous state')
        (aptitudes, best_chromosomes, previous_generation) = pickle.load(fp)

# Visualization variables
use_plotly = False
use_inline = False
animate = False

# ----------------------------- Program Start ----------------------------- 

print('Matplotlib backend:' + matplotlib.get_backend())

running_colab = 'google.colab' in str(get_ipython())
if running_colab:
  print('Running on CoLab')
else:
  print('Not running on CoLab')

# ::::: Parte 1: Generación inicial :::::
# Llenado pseudoaleatorio sin repetición
# Tratamos de trabajar con numeros de 8 bits max value 255
generation = []
if load_previous and previous_generation:
    generation = previous_generation
    best_chromosome_of_generation = generation[get_best_chromosome(generation, range(num_of_chromosomes))]
else:
    generation = [ random_chromosome(255, num_of_genes) for i in range(num_of_chromosomes) ] # Equals to the full population

# Calculate aptitudes
for j in range(num_of_chromosomes):
    generation[j]['aptitude'] = calculate_aptitude(x_range, y_range, generation[j]['values'], original_curve, weight)

print('Start of competition')
for i in range(num_of_generations):
    new_generation = []
    best_chromosome_of_generation = None
        
    for j in range(num_of_chromosomes // 2):
        for k in range(tries):
            # Select the competitors using the percentage_of_opponents defined (father)
            competitors_ids_father = default_rng().choice(num_of_chromosomes, size=int(num_of_chromosomes * percentage_of_opponents), replace=False)

            # Select the competitors using the percentage_of_opponents defined (mother)
            competitors_ids_mother = default_rng().choice(num_of_chromosomes, size=int(num_of_chromosomes * percentage_of_opponents), replace=False) 
            
            winner_father_id = get_best_chromosome(generation, competitors_ids_father)
            winner_mother_id = get_best_chromosome(generation, competitors_ids_mother)
            
            #Validate parents are not equals (to maintain diversity)
            if ((generation[winner_father_id]['values'] != generation[winner_mother_id]['values']).all()):
                break
        
        new_generation.extend(reproduce(generation[winner_father_id], generation[winner_mother_id]))    
    
    # Mutate chromosomes
    if allow_mutation:
        to_mutate_ids = default_rng().choice(num_of_chromosomes, size=int(num_of_chromosomes * mutation_percentage), replace=False)
        for id in to_mutate_ids:
            mutate(new_generation[id])
        
    # Calculate aptitudes
    for j in range(num_of_chromosomes):
        new_generation[j]['aptitude'] = calculate_aptitude(x_range, y_range, new_generation[j]['values'], original_curve, weight)
    
    # Select elitism
    if allow_elitism:
        elitism = sorted(new_generation + generation, key=lambda c : c['aptitude'])
        new_generation = elitism[:num_of_chromosomes]
        best_chromosome_of_generation = new_generation[0]
    else:
        best_chromosome_of_generation = generation[get_best_chromosome(generation, range(num_of_chromosomes))]
    
    best_chromosomes.append(best_chromosome_of_generation['values'])
    aptitudes.append(best_chromosome_of_generation['aptitude'])
    
    if use_inline:
        display.clear_output(wait = True)
        print_plots(x_range, y_range, original_mesh_curve, aptitudes, decoded_values(best_chromosome_of_generation['values'], weight), use_plotly=use_plotly, use_inline=use_inline)
        print('Generation', i + 1, 'of', num_of_generations)
        print('Best aptitude:', best_chromosome_of_generation['aptitude'])
    else:
        print('Generation', i + 1, 'of', num_of_generations)
        print('Best aptitude:', best_chromosome_of_generation['aptitude'])
    
    generation = new_generation
    if (i + 1) % check_point == 0 and save_state:
        # Store history state
        with open(history_file, 'wb') as fp:
            pickle.dump((aptitudes, best_chromosomes, generation), fp)
        print('Checkpoint stored!')

# Store history state
if save_state:
    with open(history_file, 'wb') as fp:
        pickle.dump((aptitudes, best_chromosomes, generation), fp)
    print('Checkpoint stored!')
    
print_plots(x_range, y_range, original_mesh_curve, aptitudes, decoded_values(best_chromosome_of_generation['values'], weight), use_plotly=False, use_inline=running_colab)

############### Animate ###############
if animate:    
    # Define frames
    X, Y = np.meshgrid(x_range, y_range)
    nb_frames = num_of_generations
    fig = go.Figure(frames=[go.Frame(
        data= [
            go.Surface(x=X, y=Y, z = get_mesh_curve(x_range, y_range, decoded_values(best_chromosomes[k], weight)), colorscale='viridis', opacity=0.75),
            go.Surface(z=original_mesh_curve, x=X, y=Y)
            ],
        name=str(k) # you need to name the frame for the animation to behave properly
        )
        for k in range(nb_frames)])
    
    # Add data to be displayed before animation starts
    fig.add_trace(
        go.Surface(x=X, y=Y, z = get_mesh_curve(x_range, y_range, decoded_values(best_chromosomes[0], weight)), colorscale='viridis', opacity=0.75))
     
    fig.add_trace(
        go.Surface(z=original_mesh_curve, x=X, y=Y))
    
    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},}
    
    sliders = [{
                "pad": {"b": 10, "t": 60},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [{
                        "args": [[f.name], frame_args(0)],
                        "label": str(k),
                        "method": "animate",}
                    for k, f in enumerate(fig.frames)
                ],}]
    
    # Layout
    fig.update_layout(
             title='Ocupation',
             updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(20)],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
             ],
             sliders=sliders
    )
    
    if use_inline:
        fig.show()
    else:
        fig.show(renderer="browser")