# Ground Motion prediction ACCelerated (gmacc)
This is the supplement for the publication xxxx by L. Lehmann et al.

This source code was produced during the doctoral studies of Lukas Lehmann at the University of Potsdam within the scope of the EWRICA project (funded by BMBF).

## Usage
In the examples directory, there are exemplary scripts that (01) first allow the evaluation of the Pyrocko framework to create ground motion maps. Here, the Kumamoto Earthquakes serves as example. In general, any other earthquake can be tested, as long as waveform data (as mseed), station inventory (as stationxml) and event parameter (preferably as json and if available or wanted a finite-fault file, e.g. .fsp or shakemap coordinate file). There is an instruction file that gives detailed information about the application.

Further (02) includes scripts that help to create a synthetic database and finally train a Neural Network. See the instruction file for more details.

In (03) exemplary shows how to use the trained Neural Networks. 

## Getting started
After installing this package, a central point of this package is to use the Pyrocko framework of combining source models with Green's Functions. Therefore, it is required to have a functioning GF store. Either one calculates its own GFs (see https://pyrocko.org/docs/current/apps/fomosto/) or uses/downloads an existing one (see https://greens-mill.pyrocko.org/). In general any store can be used, but depending on used velocity model and keeping the computational time, especially for the database generation, within feasible and reasonable limits, we recommend using GF sampling frequencies lower than 2Hz.

