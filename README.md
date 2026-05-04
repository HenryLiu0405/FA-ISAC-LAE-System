\* Enhancing ISAC Performance in Low-Altitude Economy with Fluid Antennas



This repository contains the MATLAB source code for the paper:

\*\*"Enhancing ISAC Performance in Low-Altitude Economy with Fluid Antennas"\*\*

accepted at the 17th International Conference on Wireless Communications and Signal Processing (WCSP 2025), Chongqing, October 23-25, 2025

\*\*Authors:\*\*Yiping Zuo, Yupeng Nie, Hengyi Liu, Lingfeng Zuo, Chen Dai


\*\*"Fluid Antenna-aided ISAC Systems for Low-Altitude Economy Networks"\*\*

accepted at IEEE Journal of Selected Topics in Signal Processing (JSTSP)

\*\*Authors:\*\*Yiping Zuo, Yupeng Nie, Weicong Chen, Chen Dai, Weibei Fan, Jun Zhang

\## 📄 Abstract



In low-altitude economic (LAE) networks, integrated sensing and communication (ISAC) exhibits transformative potential yet is constrained by fixed-antenna architectures amid dynamic environments. This project implements a \*\*Fluid Antenna (FA)\*\* aided ISAC system to address these challenges.



We formulate the beamforming and antenna positioning strategies as a non-cooperative game to maximize the total communication and sensing rate. A mixed alternating iterative algorithm based on \*\*Sine Cosine Algorithm (SCA)\*\* and \*\*Particle Swarm Optimization (PSO)\*\* (referred to as \*\*SCPSO\*\*) is developed to find the optimal solutions.



\## 🛠️ Code Structure



\* `main.m`: The main simulation script implementing the mixed SCPSO-based alternating iterative algorithm (Algorithm 3 in the paper).

\* `Modules/`: Contains standalone test scripts for SCA and PSO components.



\## 🚀 How to Run



1\.  Ensure you have MATLAB installed (Optimization Toolbox recommended).

2\.  Clone this repository or download the source code.

3\.  Open MATLAB and navigate to the folder.

4\.  Run the `main` function:

&nbsp;   ```matlab

&nbsp;   main

&nbsp;   ```



\## 📊 Parameters



Key simulation parameters (defined in `main.m`):

\* \*\*K (BSs)\*\*: 3

\* \*\*M (Antennas)\*\*: 4

\* \*\*N (Users)\*\*: 2

\* \*\*L (Paths)\*\*: 12



\## 🔗 Citation



If you find this code useful for your research, please cite our paper: 

Y. Zuo, Y. Nie, H. Liu, L. Zuo and C. Dai, "Enhancing ISAC Performance in Low-Altitude Economy with Fluid Antennas," 2025 Seventeenth International Conference on Wireless Communications and Signal Processing (WCSP), Chongqing, China, 2025, pp. 1-6, doi: 10.1109/WCSP68525.2025.1010111. keywords: {Wireless communication;Fluids;Array signal processing;Simulation;Transmitting antennas;Signal processing algorithms;Games;Integrated sensing and communication;Antennas;Convergence;Fluid antenna;sensing;communication;LAE},

Y. Zuo et al., "Fluid Antenna-aided ISAC Systems for Low-Altitude Economy Networks," in IEEE Journal of Selected Topics in Signal Processing, doi: 10.1109/JSTSP.2026.3671173.
keywords: {Integrated sensing and communication;Antennas;Autonomous aerial vehicles;Array signal processing;Optimization;Resource management;Numerical models;Vehicle dynamics;Reconfigurable intelligent surfaces;Iterative methods;Fluid antenna;sensing;communication;LAE},







