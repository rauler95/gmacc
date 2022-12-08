# Creation of a compariosn between PWS (pyrocko-waveform-simulation) and observational data

To execute a calculation of an event specify the parameters in a config file, see `example_event_config.yaml`.
And, if the name changed, change it in the script:

*event_ground_motion_map_generation.py*

Important for the execution is the existence of waveform, station and event file(s), if sourcemode is RS than also a faultfile (preferably in .fsp format). In general see example even Kumamoto directory. For more details about the configuration look in `gmacc/src/config.py`.

In the end, several file, mainly maps, are produced in the given output directory.