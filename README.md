\# Enhancing ISAC Performance in Low-Altitude Economy with Fluid Antennas



This repository contains the MATLAB source code for the paper:

\*\*"Enhancing ISAC Performance in Low-Altitude Economy with Fluid Antennas"\*\*



\*\*Authors:\*\*Yiping Zuo, Yupeng Nie, Hengyi Liu, Lingfeng Zuo, Chen Dai



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

( the information will be released when the paper is published)

