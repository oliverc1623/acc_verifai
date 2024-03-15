# Installation of VerifAI and Scenic

1. Clone the [VerifAI](https://github.com/BerkeleyLearnVerify/VerifAI) repository and [Scenic](https://github.com/BerkeleyLearnVerify/Scenic).
2. Checkout the `kesav-v/multi-objective` branch of VerifAI and the `kesav-v/multi-objective` branch of Scenic.
3. Install `poetry` if you haven’t already done so.
4. Run `poetry shell` from the VerifAI repo and make sure it spawns an environment with Python 3.8+.
5. Run `poetry install`.
6. Go to the location where `Scenic` was cloned and run `poetry install` (while in the same environment that was used for VerifAI).
7. Any other missing packages when running the falsifier script can be installed using `pip`.

# Running a simulator

1. Download and install [LGSVL](https://www.svlsimulator.com/) or [Carla](https://carla.org/) and its respective Python APIs, according to your prefferences or needs.
2. Open the desired simulator and start an API simulation
3. Run the bash script `script.sh`, by default it runs LGSVL, but you can change the parameters of its commands from `--model lgsvl` to `--model carla`
