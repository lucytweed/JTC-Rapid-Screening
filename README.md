This is a python programme that calculates pressure buildup and degree of Joule Thomson cooling that occur during injection of CO2 into a depleted gas reservoir. 

The model solves a coupled set of PDEs for pressure and temperature in a radially symmetric and infintely acting reservoir. It is therefore suitable for investigating the early stages of injection. The model assumptions and procedure are described in detail in Tweed et al., 2024, Journal of Fluid Mechanics.

To utilise the tool carry out the following steps:
  1. Download the files to a directory on your computer.
  2. Copy and rename the input_parameters.xlsx data file template. Fill this out for the reservoir and injection conditions of interest. All options and variables are documented within the file.
  3. Run the code from the command line with the command "python JTC.py"
  4. Follow the prompts in the command line
  5. The model output will be saved to a directory called "JTC_creening_runs" within the parent directory. 

A few of the options for running the model are dependent on the Coolprop python library. This is a library that allows automatic calculation of CO2 thermophysical properties. To activate these options the Coolprop python library must first be installed. Installation instructions can be found here:
http://www.coolprop.org/coolprop/wrappers/Python/index.html#automatic-installation 
