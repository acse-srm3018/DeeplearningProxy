# Deep learning Proxy of reservoir simulation

## Project description
The common modelling method for reservoir fluid flow model is to use a finite-difference (FD). Since FD uses computations on a discrete computational grid, and its precision is strongly related on a grid resolution. Being accurate, fluid flow simulation through porous media may be really time consuming (several hours for a single run) and we have to perform hundreds of thousands of run to calibrate the models using observation data and to correctly quantifying uncertainty.  Therefore, it is almost impossible to run iterative optimization and history-matching methods using this kind of models in sufficient time. There are many techniques aimed to speed up the simulation process: starting from simple downscaling and local grid refinement and ending with data-driven Reduced Order Modelling (ROM) approaches.
In oil industry a fast and accurate model are required for three-phase flow simulation on a 3D computational grid. All conventional approaches are unable to solve this task properly: either they hardly produce solutions for spatial fields (pressure, saturation, etc.) or they can work only under nonrealistic greatly simplified settings. In this project. 
We propose to employ deep learning (DL) methods to create an efficient proxy of the 3D fluid flow simulator. The objective is to train a DL model that will take 3D cells of permeability and porosity values as an input and then produce 3D images of saturation and pressure as a function of time with given discretization.
Fully-convolutional neural networks can represent a way to scale the proposed approach to a whole oilfield. Experiments with whole and real cases are retained for future work. Proposed meta modelling approach is based on the idea of forecasting in latent variable space. The same idea was used in to approximate the dynamics of an environment with high-dimensional state space for model-based.


## References
Turgay Ertekin, J.H. Abou-Kassem, and G.R. King.Basic Applied Reservoir Simulation. SPE Textbook SeriesVol. 7, 2001.
