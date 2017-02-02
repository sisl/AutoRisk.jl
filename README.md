# AutoRisk

[![Build Status](https://travis-ci.org/wulfebw/AutoRisk.jl.svg?branch=master)](https://travis-ci.org/wulfebw/AutoRisk.jl)

[![Coverage Status](https://coveralls.io/repos/wulfebw/AutoRisk.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/wulfebw/AutoRisk.jl?branch=master)

[![codecov.io](http://codecov.io/github/wulfebw/AutoRisk.jl/coverage.svg?branch=master)](http://codecov.io/github/wulfebw/AutoRisk.jl?branch=master)

<!-- MarkdownTOC -->

- [quickstart](#quickstart)
    - [install](#install)
    - [collection](#collection)
- [package outline](#package-outline)
    - [src](#src)
        - [generation](#generation)
            - [roadway](#roadway)
            - [scene](#scene)
            - [behavior](#behavior)
        - [evaluation](#evaluation)
        - [collection](#collection-1)
        - [analysis](#analysis)
    - [scripts](#scripts)
    - [test](#test)
    - [data](#data)
    - [tests](#tests)

<!-- /MarkdownTOC -->

# quickstart

## install
- First install the julia components of AutoRisk
```julia
Pkg.clone("https://github.com/wulfebw/AutoRisk.jl.git")
Pkg.build("AutoRisk")

# check that AutoRisk can be imported
using AutoRisk

# check that all the tests pass
Pkg.test("AutoRisk")
```

## collection
- cd scripts/collection and run:
```bash
julia run_collect_heuristic_dataset.jl --num_scenarios 1 --num_monte_carlo_runs 1
```
- This should generate a dataset from a single scenario (i.e., a roadway, scene, behavior model tuple), output some information about that dataset, and save it to a file in the data directory.

# package outline
The goal of this package, "AutoRisk", is to make it easy to collect datasets focused on automotive risk. It builds on AutomotiveDrivingModels.jl

## src

### generation
- Source code in this subdirectory deals with the generation of roadways, scenes, and behavior.

#### roadway
- In performing a simulation, the first decision to make is what roadway should be used in the simulation. 
    + Currently two roadways are considered, a stadium roadway and a straight roadway.

#### scene
- Scene generation in a simulation context entails populating a roadway with vehicles, each with their own state (i.e., position, velocity, acceleration, heading, etc).
    + The currently implemented heuristic scene generation populates a roadway using a set of heuristic rules.

#### behavior
- Behavior generation deals with populating a scene with driver models that have different types of behavior.
    + The simplest case is that of generating static parameter values for heuristic driver models.
    + Slightly more complex is sampling parameter values in an either correlated or uncorrelated manner for those heuristic models.

### evaluation
- The evaluation subdirectory deals with evaluating the safety or risk associated with a generated scene. This can be viewed as a form of "policy evaluation", where the policy is defined by the driver models. The goal is to give risk estimates (where each risk estimate takes the form of a probability distribution over some event like e.g., hard braking), and we perform that here by simulating a scene many times and computing aggregate statistics from those samples.

### collection
- This subdirectory contains code that orchestrates the collection of datasets. Specifically, there is a type of that controls the interaction between the generators and evaluators previously discussed, a type that runs collectors in parallel across multiple processes, and a type that acts as a container for the collected data.

### analysis
- Each roadway, scene, behavior, and evaluation grouping is uniquely identified by a seed associated with it. Because of this, given a seed, we can visualize what the scene looks like and how evaluation of it proceeds. This directory deals with this process of analyzing the scenes and how they are evaluated, and for now just contains some evaluation code.

## scripts 
- This directory contains scripts for using AutoRisk to collect datasets (collection), for fitting models to those datasets (compression and prediction), and for visualizing datasets and scenarios (visualization).

## test
- This directory contains test for the AutoRisk source files. To run tests:
```bash
julia runtests.jl
```

## data
- This directory holds data associated with the project such as generated datasets, log files, neural network files (useable from julia), snapshots and summaries (files saved by tensorflow during training), and visualizations (generally the output from the visualization scripts).

## tests
- This directory contains tests for the python code in the scripts directory. To run the tests:
```bash
python run_tests.py
```
