# Installation of VerifAI and Scenic

1. Clone the [VerifAI](https://github.com/BerkeleyLearnVerify/VerifAI) repository and [Scenic](https://github.com/BerkeleyLearnVerify/Scenic) version 2.1.0.
2. Use python 3.8, higher versions of python might produce conflicts within some of the used libraries. 
3. Install both repositories, first Scenic then VerifAI. Go to their folders and run `python -m pip install -e` (we recommend installing everything in a virtual enviroment)
4. Download [Carla](https://carla.org/) (versions 0.9.12-0.9.15 work) 
5. Set the enviromental variables of carla and its wheel python file.
6. Our experiments use `Town06` so make sure you install the additional maps for Carla.
7. Download this repository.

# Running the experiments

1. Activate the virtual environment where Scenic and VerifAI are installed.
2. Open Carla simulator 
3. To run an experiment run the python script `falsifier.py` with parameters: `--model carla` `--path path/to/scenario` eg: `--path scenarios-ddas/persistent_attack.scenic` `--e output_name` (to name the file where the falsification table will be stored) `--route folder` (to create a new folder to save the results of your simulation

# Additional notes

- Take into account the variable `LEAD_TO_EGO` in the scenario to specify the setpoint distance and make sure it matches the variable self.d in the acc.py file (this will be fixed)
- Remember to change the variable `verifaiSampleType` in each scenario, our experiments have testest `bo`(Bayesian Optimization)  and `ce` (Cross Entropy) as of Feb/2025
