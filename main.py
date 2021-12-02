# -*- coding: utf-8 -*-

import sys
import matplotlib
import matplotlib.pyplot as plt
from IPython import display
from numpy.random import randint, default_rng

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
    Z = z[X,Y]
    # Get labels
    x_labels, y_labels = data.axes
    return (x_range, y_range, x_labels, y_labels, X, Y, Z)

# Function to retrieve the generated curve
def curve(x_range, y_range, c):
    points = np.zeros((len(x_range), len(y_range)))
    X, Y = np.meshgrid(x_range, y_range)
    
    # TODO: should this values come from chromosome?
    ##################################################
    m1=c[0];
    m2=c[1];
    m3=c[2];
    m4=c[3];
    m5=c[4];
    m6=c[5];
    de1=c[6] or 0.01;
    de2=c[7] or 0.01;
    de3=c[8] or 0.01;
    de4=c[9] or 0.01;
    de5=c[10] or 0.01;
    de6=c[11] or 0.01;
    ##################################################
    
    p1=c[12];
    p2=c[13];
    p3=c[14];
    p4=c[15];
    p5=c[16];
    p6=c[17];
    p7=c[18];
    p8=c[19];
    p9=c[20];
    
    q1=c[21];
    q2=c[22];
    q3=c[23];
    q4=c[24];
    q5=c[25];
    q6=c[26];
    q7=c[27];
    q8=c[28];
    q9=c[29];
    
    r1=c[30];
    r2=c[31];
    r3=c[32];
    r4=c[33];
    r5=c[34];
    r6=c[35];
    r7=c[36];
    r8=c[37];
    r9=c[38];

    for x in x_range:
        x_temp = x / x_range[-1];
        for y in y_range:
            y_temp = y / y_range[-1];
            
            mf1 = np.exp(( -(x_temp - m1)**2) / (2 * de1**2))
            mf2 = np.exp(( -(x_temp - m2)**2) / (2 * de2**2))
            mf3 = np.exp(( -(x_temp - m3)**2) / (2 * de3**2))
            mf4 = np.exp(( -(y_temp - m4)**2) / (2 * de4**2))
            mf5 = np.exp(( -(y_temp - m5)**2) / (2 * de5**2))
            mf6 = np.exp(( -(y_temp - m6)**2) / (2 * de6**2))
              
            inf1 = mf1 * mf4
            inf2 = mf1 * mf5
            inf3 = mf1 * mf6
            inf4 = mf2 * mf4
            inf5 = mf2 * mf5
            inf6 = mf2 * mf6
            inf7 = mf3 * mf4
            inf8 = mf3 * mf5
            inf9 = mf3 * mf6
              
            reg1 = inf1 * (p1 * x_temp + q1 * y_temp + r1)
            reg2 = inf2 * (p2 * x_temp + q2 * y_temp + r2)
            reg3 = inf3 * (p3 * x_temp + q3 * y_temp + r3)
            reg4 = inf4 * (p4 * x_temp + q4 * y_temp + r4)
            reg5 = inf5 * (p5 * x_temp + q5 * y_temp + r5)
            reg6 = inf6 * (p6 * x_temp + q6 * y_temp + r6)
            reg7 = inf7 * (p7 * x_temp + q7 * y_temp + r7)
            reg8 = inf8 * (p8 * x_temp + q8 * y_temp + r8)
            reg9 = inf9 * (p9 * x_temp + q9 * y_temp + r9)
              
            b = inf1 + inf2 + inf3 + inf4 + inf5 + inf6 + inf7 + inf8 + inf9
            a = reg1 + reg2 + reg3 + reg4 + reg5 + reg6 + reg7 + reg8 + reg9
            
            points[x, y] = (a / b)

    return points[X, Y]

# Returns the chromsomes values decoded by w
def decoded_values(chromosome_values, w):
    return [x / 255 for x in chromosome_values[:6]] \
        + [x / 25.5 for x in chromosome_values[6:12]] \
        + [x / w for x in chromosome_values[12:]]

# Sets the aptitude value of a chromosome (Area between the curve and the original curve)
def calculate_aptitude(x_range, y_range, chromosome, original_curve, w = 1):
    decoded = decoded_values(chromosome['values'], w)
    aprox_curve = curve(x_range, y_range, decoded)
    chromosome['aptitude'] = np.abs(original_curve - aprox_curve).sum()
    
# Returns the best chromosome from the competitors (Minimum sum of distances)
def get_best_chromosome(population, competitors_ids):
    return min(competitors_ids, key=lambda chromosome_id: population[chromosome_id]['aptitude'])

# Generates a random filled chromosome 
def random_chromosome(max_value, num_of_genes):
    return {'values': list(default_rng().choice(max_value + 1, size=num_of_genes, replace = True).astype(int)),
            'aptitude': 0}

# Reproduce two chromosomes
def reproduce(chromosome_father, chromosome_mother):
    # Spliting by 8 bits per value
    father = chromosome_father['values'][:]
    mother = chromosome_mother['values'][:]
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
        {'values': child1, 'aptitude': 0}, 
        {'values': child2, 'aptitude': 0}]

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
            values[position] = 1
        
        mask = 1 << mod_partition
        values[position] = values[position] ^ mask
    chromosome['values'] = values


############### Read Original Data ###############
import pandas as pd
import numpy as np
(x_range, y_range, x_labels, y_labels, X, Y, Z) = data_curve()

# ----------------------------- Variable Initialization ----------------------------- 
num_of_chromosomes = 1000           # Equals to the number of chromosomes per population
num_of_generations = 100            # Number of generations to reproduce
num_of_genes = 39                   # Equals to the number of genes
percentage_of_opponents = 0.05      # Percentage of chromosomes from the population to compite
weight = 255 / Z.max()              # To set a max value of our data
tries = 10

# Mutation/Elitism variables
allow_mutation = True
mutation_percentage = 0.1
allow_elitism = True

# History variables
aptitudes = []
best_chromosomes = []

# Visualization variables
use_inline = True

# ----------------------------- Program Start ----------------------------- 

print('Matplotlib backend:' + matplotlib.get_backend())

# ::::: Parte 1: Generación inicial :::::
# Llenado pseudoaleatorio sin repetición
# Tratamos de trabajar con numeros de 8 bits max value 255
random_chromosome(255, num_of_genes)
generation = [ random_chromosome(255, num_of_genes) for i in range(num_of_chromosomes) ] # Equals to the full population

# Calculate aptitudes
for j in range(num_of_chromosomes):
    calculate_aptitude(x_range, y_range, generation[j], Z, weight)

print('Start of competition')
for i in range(num_of_generations):
    print('Running generation', i + 1, 'of', num_of_generations)
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
            if (generation[winner_father_id] != generation[winner_mother_id]):
                break
        
        new_generation.extend(reproduce(generation[winner_father_id], generation[winner_mother_id]))    
    
    # Mutate chromosomes
    if allow_mutation:
        to_mutate_ids = default_rng().choice(num_of_chromosomes, size=int(num_of_chromosomes * mutation_percentage), replace=False)
        for id in to_mutate_ids:
            mutate(new_generation[id])
        
    # Calculate aptitudes
    for j in range(num_of_chromosomes):
        calculate_aptitude(x_range, y_range, new_generation[j], Z, weight)
    
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
        best_curve = curve(x_range, y_range, decoded_values(best_chromosome_of_generation['values'], weight))
        
        matplotlib.use('module://matplotlib_inline.backend_inline')
        
        #fig = plt.figure()
        fig = plt.figure(figsize = (18, 8))
        
        #ax = fig.gca(projection='3d')
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.set_xlabel('Fecha')
        ax1.set_xticklabels(x_labels)
        ax1.set_ylabel('')
        ax1.set_yticks(y_range)
        ax1.set_yticklabels(y_labels)
        ax1.set_zlabel('Population %')
    
        # Plot the surface.
        surf = ax1.plot_surface(X, Y, Z, cmap=plt.cm.viridis, alpha=1)
        surf_aprox = ax1.plot_surface(X, Y, best_curve, cmap=plt.cm.inferno, alpha=0.5)
        
        ax2 = fig.add_subplot(1,2,2)
        ax2.clear()
        ax2.plot(aptitudes, linewidth=2, color='royalblue', alpha=0.5, label='Aptitude of generation' + str(i + 1))
        ax2.grid(linestyle='--')
        ax2.legend()
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Aptitude')
        ax2.title.set_text("Aptitude: {:.2f}".format(aptitudes[i]))
        
        # Add a color bar which maps values to colors.
        #fig.colorbar(surf, shrink=0.5, aspect=5)
        display.clear_output(wait = True)
        
        
        plt.show()
        print('Generation', i + 1, 'of', num_of_generations)
        print('Best:', best_chromosome_of_generation)
        print('Decoded:', decoded_values(best_chromosome_of_generation['values'], weight))
    
    generation = new_generation

sys.exit(0)

############### Plot Data ###############
use_plotly = False
use_inline = False

if use_plotly:
    import plotly.graph_objects as go
    fig = go.Figure(data=[
        go.Surface(z=best_curve, x=X, y=Y, colorscale="inferno", opacity=0.5),
        go.Surface(z=Z, x=X, y=Y, colorscale="viridis", opacity=1)
        ])
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
    fig.show(renderer="browser")
else:
    if use_inline:
        matplotlib.use('module://matplotlib_inline.backend_inline')
    else:
        matplotlib.use('Qt5Agg')

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('Fecha')
    ax.set_xticklabels(x_labels)
    
    ax.set_ylabel('')
    ax.set_yticks(y_range)
    ax.set_yticklabels(y_labels)
    
    ax.set_zlabel('Population %')

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.viridis, alpha=1)
    surf_aprox = ax.plot_surface(X, Y, best_curve, cmap=plt.cm.inferno, alpha=0.5)
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
