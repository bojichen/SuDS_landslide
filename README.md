This dataset supports the research paper titled 'Exploring the Effectiveness, Risks, and Strategies for Implementing Sustainable Drainage Systems in Landslide-Prone Catchments'. It includes code and data necessary for running simulations and analyzing results.

Three files in each folder should be run sequentially:
1. Warmup.ipynb: This file is used for the warm-up of the Modflow model. The annual average precipitation and evapotranspiration are used to achieve a steady state of the modeling domain.
2. SWMM_Modflow.py: This file couples SWMM and Modflow, representing the first year since the Sustainable Drainage Systems (SuDS) have been implemented.
3. SWMM_Modflow_01.py: This file couples SWMM and Modflow, representing the second year since the SuDS were implemented. By this year, groundwater has reached a stable state. Simulation for the third year is not necessary.
