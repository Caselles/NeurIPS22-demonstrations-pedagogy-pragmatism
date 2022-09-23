# Pragmatically Learning from Pedagogical Demonstrations in Multi-Goal Environments

### Abstract

Learning from demonstration methods usually leverage close to optimal demonstrations to accelerate training. By contrast, when demonstrating a task, human teachers deviate from optimal demonstrations and pedagogically modify their behavior by giving demonstrations that best disambiguate the goal they want to demonstrate. Analogously, human learners excel at pragmatically inferring the intent of the teacher, facilitating communication between the two agents. These mechanisms are critical in the few demonstrations regime, where inferring the goal is more difficult. In this paper, we implement pedagogy and pragmatism mechanisms by leveraging a Bayesian model of goal inference from demonstrations. We highlight the benefits of this model in multi-goal teacher-learner setups with two artificial agents that learn with goal-conditioned Reinforcement Learning. We show that combining a pedagogical teacher and a pragmatic learner results in faster learning and reduced goal ambiguity over standard learning from demonstrations, especially in the few demonstrations regime.

### Video explaining the paper

Here is a link to an illustration video for the paper: https://youtu.be/V4n16IjkNyw.

### This repository contains the code needed to reproduce the experiments in the paper.

In order to replicate our main experiments, after modifying all relevant paths in the scripts:

1. Train a naive and a pedagogical teacher with train_teacher.py

command : mpirun -n 24 python train_teacher.py --cuda --pedagogical-teacher True or False

2. Generate a dataset of demonstrations with generate_teacher_demo_dataset_mpi.py

command : mpirun -n 24 python generate_teacher_demo_dataset_mpi.py

3. Train the learner with train_learner.py

command : mpirun -n 24 python train_learner.py --cuda --learner-from-demos True --teacher-mode pedagogical or naive --pragmatic-learner True or False --sqil True  --compute-statistically-significant-results True --predictability True --reachability True

4. Create plots with plots_v2.py

command : python plots_v2.py



The experiments require a 24 core CPU. 
