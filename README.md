# AUTOMATIC-ANALYSIS-OF-BUILDING-FOOTPRINTS-WITH-VERY-HIGH-RESOLUTION-SATELLITE-IMAGERY
The topic includes:   Collection of very high resolution (VHR) satellite imagery (with less than 5 meters resolution)    
Doing some scripting/programming in Python, in order to: 
- Load the imagery; 
- Estimating building footprints and statistics with different methods; 
- Make some information, derivate work from the input.

During the project, the problem of object detection approached a kind-of novel, bottom-up simulation method:
- Raster cells are converted to numpy arrays
- Neighbor relations between cells can be created by: Moore or von Neumann neighboorhod models
- Similar (band values) and adjacent cells should be connected to a group of cells (set):
  - Group cohesion can be valued (e.g. similarity index, RMSE)
- Group functions (called on groups during iterations):
  - Area (number of cells)
  - Inner cells / Outer cell (can be calculated with neighbor counts)
  - Group perimeter: Can be derived from outer cells
  - Include holes? (Holes number)
  - Perimeter keyponts, angle derviations   
  - Neighbor groups etc...

By iteratively calling group functions, the groups can be connected to each other.
Buildings can be queried from gropus: 
- By mean RGB value, area, angles (space syntax - grammar, language processors?)

Try to use less amount of Python packages in the long-term-reseach:
- *numpy* (thats the base): numpy uses Cython calls (fastened with C)
- and if other packages avoided numpy can be toward fastened with *numba* package which can use GPU 
- for data visualisation *matplotlib* package advised (in another file)

Used structures:
- vectors, matrices, arrays, sets, dictionaries, functions (maybe classes)
- if the problem handled in a 'vectorized' manner (less class) faster evaluations garanteed
