from logic_gates import AND
from logic_gates import OR
from logic_gates import NOT
from logic_gates import XOR

and_gate = AND()
or_gate = OR()
not_gate = NOT()
xor_gate = XOR()

print(and_gate(False, False))
print(and_gate(False, True))
print(and_gate(True, False))
print(and_gate(True, True))
print(or_gate(False, False))
print(or_gate(False, True))
print(or_gate(True, False))
print(or_gate(True, True))
print(not_gate(False))
print(not_gate(True))
print(xor_gate(False, False))
print(xor_gate(False, True))
print(xor_gate(True, False))
print(xor_gate(True, True))

