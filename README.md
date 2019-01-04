# ecgsyn-python [![](https://img.shields.io/appveyor/ci/gruntjs/grunt.svg)](https://github.com/lingfliu/ecgsyn-python)

A python implementation of ecgsyn from physionet

Original reference: 

[ECGSYN: A realistic ECG waveform generator](https://www.physionet.org/physiotools/ecgsyn/)

Code implemented in Python 3.6

Dependencies:
- numpy: 1.15
- scipy: 1.10

Modifications: 

- replaced ide45 with ideint
- converted indice from matlab format (starting from 1) to python form (starting from 0)

Usage: 

To generate by default parameters, simply run:

```python
(sig, idx) = ecgsyn()
```

where sig is the generated ECG sequences, and idx is the PQRST peak indice sequence marked as [0,1,2,3,4] respectively (-1 for others).



