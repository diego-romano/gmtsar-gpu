__INSTRUCTIONS FOR INSTALLING AND RUNNING gmtsar-gpu__
----------------------------------------------

This toolset is intended to speed up InSAR processing by exploiting GPGPU through CUDA. 
Executables can substitute original sequential versions in GMTSAR. 
At the moment, only _xcorr-gpu_ has been released. Soon, _esarp-gpu_ will follow.

__REQUIREMENTS__

1) A CUDA gpu and a full CUDA installation

2) A working GMTSAR installation for full processing

__INSTALL__

1) Go to the GMTSAR github and follow the instructions to install.
       https://github.com/gmtsar/gmtsar

2) Go to the gmtsar-gpu directory and check _config.mk_ for proper paths.

3) To build the executable, type

       make

4) Replace the original _xcorr_ executable in the _bin_ directory ogf GMTSAR with _xcorr-gpu_ executable changing its name.

5) test the command _xcorr_

__RUN__

Use as described in GMTSAR documentation.
