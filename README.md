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
3. Run the python script `falsifier.py` with `--model carla` to connect to the Carla API
