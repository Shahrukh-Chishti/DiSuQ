# master execution file
import os

print('Operator definition')
os.system('./operators.py')

print('Consistency of quantization Sub-rountine')
os.system('consistency/backend.py')
os.system('consistency/basis.py')
os.system('consistency/eigensolvers.py')
os.system('consistency/precision.py')

print('Comparison : DiSuQ v/s scQubits')
os.system('comparison/scQ-fluxonium.py')
os.system('comparison/scQ-transmon.py')

print('Optimization')
os.system('optimization/spectrum-tuning.py')
os.system('optimization/box-qutrit-projection.py')
# os.system('optimization/initialization-parallelism.py') # failing

print('Multi-GPU')
os.system('multi-GPU/distributed-test.py')
os.system('multi-GPU/fluxonium-tuning.py')
